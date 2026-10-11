"""M4R margins: stance re-targeting, pitcher-to-robot clearance and the robustness sweep.

Simulation only. These cover the pure parts that the 1 kHz replays rely on; the replays
themselves (``run_m4.py``, ``m4_sweep.py``) are release evidence, not unit tests.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFICATION = REPO / "barista" / "verification"
TRAJECTORY = REPO / "barista" / "trajectory" / "m4_spiral_pour.json"


def _load(stem: str):
    name = f"barista_verification_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, VERIFICATION / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- re-targeting
def test_stance_offset_is_identity_at_base_start():
    retarget = _load("m4_retarget")
    offset = retarget.StanceOffset((0.1, 0.2, 0.0), (0.1 + 5e-7, 0.2, 5e-7))
    assert offset.is_identity
    assert not retarget.StanceOffset((0.1, 0.2, 0.0), (0.1, 0.2, math.radians(0.5))).is_identity
    assert not retarget.StanceOffset((0.1, 0.2, 0.0), (0.102, 0.2, 0.0)).is_identity


def test_stance_offset_carries_a_rigidly_moved_point_back():
    retarget = _load("m4_retarget")
    nominal = (-0.0187, 0.0955, 0.0)
    dyaw = math.radians(-2.0)
    measured = (nominal[0] + 0.002, nominal[1] - 0.001, dyaw)
    offset = retarget.StanceOffset(nominal, measured)
    # A point fixed to the robot, moved with it from the nominal stance to the measured one.
    point_nominal = np.array([0.0934, 0.32, 0.18])
    rotation = retarget.rot_z(dyaw)
    point_measured = point_nominal.copy()
    point_measured[:2] = (
        rotation[:2, :2] @ (point_nominal[:2] - np.array(nominal[:2])) + np.array(measured[:2])
    )
    np.testing.assert_allclose(offset.to_nominal_point(point_measured), point_nominal,
                               atol=1e-12)
    np.testing.assert_allclose(offset.to_nominal_rotation(rotation), np.eye(3), atol=1e-12)
    assert offset.as_dict()["dyaw_deg"] == pytest.approx(-2.0)


def test_planar_pose_reads_yaw_from_the_rotation():
    retarget = _load("m4_retarget")
    x, y, yaw = retarget.planar_pose([1.0, 2.0, 3.0], retarget.rot_z(0.3))
    assert (x, y) == (1.0, 2.0) and yaw == pytest.approx(0.3)


def test_rot_log_of_a_z_rotation():
    retarget = _load("m4_retarget")
    np.testing.assert_allclose(retarget.rot_log(retarget.rot_z(0.2)), [0, 0, 0.2], atol=1e-12)
    np.testing.assert_allclose(retarget.rot_log(np.eye(3)), np.zeros(3), atol=1e-12)


# --------------------------------------------------------------------------- clearance
def test_pitcher_robot_clearance_summary_takes_the_carried_minimum():
    run_m4 = _load("run_m4")
    rows = [
        {"phase": "lift_hold", "time_s": 5.0, "pitcher_robot_clearance_m": 0.0,
         "pitcher_robot_nearest_pair": "ignored;not-carried"},
        {"phase": "transit", "time_s": 9.0, "pitcher_robot_clearance_m": 0.030,
         "pitcher_robot_nearest_pair": "robot/LF_forearm_link_meshcol;bar_handle_upper"},
        {"phase": "spiral", "time_s": 38.2, "pitcher_robot_clearance_m": 0.006,
         "pitcher_robot_nearest_pair": "robot/display_module_link_meshcol;bar_spout_tip"},
        {"phase": "spiral", "time_s": 30.0, "pitcher_robot_clearance_m": 0.012,
         "pitcher_robot_nearest_pair": "robot/display_module_link_meshcol;bar_pitcher_body"},
        {"phase": "final_settle", "time_s": 56.0, "pitcher_robot_clearance_m": math.inf,
         "pitcher_robot_nearest_pair": ""},
    ]
    summary = run_m4._pitcher_robot_clearance_summary(rows)["by_scope"]
    assert summary["carried"] == {
        "minimum_m": 0.006, "nearest_pair": "robot/display_module_link_meshcol;bar_spout_tip",
        "time_s": 38.2,
    }
    assert summary["transit"]["minimum_m"] == 0.030
    assert summary["spiral"]["time_s"] == 38.2
    assert "lift_hold" not in summary and "pour_in" not in summary


def test_clearance_excludes_only_the_claw_and_the_water_proxies():
    run_m4 = _load("run_m4")
    assert run_m4.CLAW_PREFIXES == ("robot/LF_palm", "robot/LF_finger")
    assert "robot/display_module_link_meshcol".startswith(run_m4.CLAW_PREFIXES) is False
    assert "robot/LF_forearm_link_meshcol".startswith(run_m4.CLAW_PREFIXES) is False
    assert "bar_spout_tip".endswith(run_m4.PITCHER_INTERNAL_GEOMS) is False
    assert set(run_m4.PITCHER_INTERNAL_GEOMS) < set(run_m4.EXPECTED_PITCHER_NONCOLLIDING)


# --------------------------------------------------------------------------- rate / parser
def test_pair_clearance_exempts_only_exact_grasp_pairs_and_keeps_penetration():
    module = _load("m4_pair_clearance")
    names = {
        0: "robot/LF_finger_grip_insert_link_meshcol",
        1: "robot/LF_palm_grip_insert_link_meshcol",
        2: "robot/LF_palm_pad_b_link_meshcol",
        3: "bar_top_grip_collar_a", 4: "bar_top_grip_collar_b",
        5: "bar_pitcher_body", 6: "robot/display_module_link_meshcol",
    }
    tracker = module.PairClearance([0, 1, 2, 6], [3, 4, 5], names)
    model = SimpleNamespace(geom_rbound=np.zeros(7))
    data = SimpleNamespace(geom_xpos=np.zeros((7, 3)), ncon=1,
                           contact=[SimpleNamespace(geom1=3, geom2=0, dist=-0.001)])
    distances = {(0, 3): 0.003, (1, 4): 0.0, (2, 4): 0.002, (6, 5): 0.006}
    def distance(a, b):
        assert (a, b) != (0, 3), "real contact must not be sent to the confirmation solver"
        return distances.get((a, b), 0.05)
    tracker.observe(model, data, 22.0, distance, 0.1)
    result = tracker.report()
    assert result["unrestricted"]["minimum_m"] == -0.001
    excluded = result["excluding_intentional_grasp_pairs"]
    assert excluded["minimum_m"] == 0.002
    assert excluded["nearest_pair"] == [names[2], names[4]]
    assert len(result["excluded_pairs"]) == 4
    physical = result["physical_collision_contacts"]
    assert [row["pair"] for row in physical] == [list(p) for p in module.INTENTIONAL_GRASP_PAIRS]
    assert physical[0]["minimum_m"] == -0.001
    assert physical[0]["maximum_normal_force_n"] is None
    assert physical[1]["contact_count"] == 0
    # A different pitcher part touching the exempt finger is still included.
    data.contact = [SimpleNamespace(geom1=0, geom2=5, dist=-0.0002)]
    tracker.observe(model, data, 23.0, lambda a, b: 0.05, 0.1)
    assert tracker.report()["excluding_intentional_grasp_pairs"]["minimum_m"] == -0.0002
    assert tracker.report()["non_exempt_actual_contact"]["minimum_m"] == -0.0002


@pytest.mark.parametrize("robot,pitcher,interface", [
    ("robot/LF_finger_grip_insert_link_meshcol", "bar_top_grip_collar_a", "F"),
    ("robot/LF_finger_grip_insert_link_visual", "bar_top_grip_collar_a", "F"),
    ("robot/LF_palm_grip_insert_link_meshcol", "bar_top_grip_collar_b", "P"),
    ("robot/LF_palm_grip_insert_link_visual", "bar_top_grip_collar_b", "P"),
])
def test_exact_interface_aliases_do_not_establish_grip(robot, pitcher, interface):
    module = _load("m4_pair_clearance")
    tracker = module.PairClearance([0], [1], {0: robot, 1: pitcher})
    model = SimpleNamespace(geom_rbound=np.zeros(2))
    data = SimpleNamespace(geom_xpos=np.zeros((2, 3)), ncon=0, contact=[])
    tracker.observe(model, data, 1.0, lambda a, b: -0.002, 0.1)
    report = tracker.report()
    assert report["interface_contract_version"] == "m4r-two-interface-v1"
    assert report["unrestricted"]["minimum_m"] == -0.002
    assert report["excluding_intentional_grasp_pairs"]["minimum_m"] == math.inf
    assert all(row["contact_count"] == 0 for row in report["physical_collision_contacts"])
    mapping = next(row for row in report["intentional_interfaces"] if row["interface"] == interface)
    assert [robot, pitcher] in (mapping["collision_pair"], mapping["visual_alias_pair"])


@pytest.mark.parametrize("robot,pitcher", [
    (f"robot/LF_{side}_grip_insert_link_{representation}", part)
    for side in ("finger", "palm")
    for representation in ("meshcol", "visual")
    for part in ("bar_top_grip_safety_shelf", "bar_top_grip_bar", "bar_top_grip_strut_a",
                 "bar_top_grip_strut_b", "bar_pitcher_body",
                 "bar_top_grip_collar_b" if side == "finger" else "bar_top_grip_collar_a")
] + [
    ("robot/LF_palm_link_visual", "bar_top_grip_collar_b"),
    ("robot/LF_finger_link_visual", "bar_top_grip_collar_a"),
    ("robot/LF_palm_pad_b_link_meshcol", "bar_top_grip_collar_b"),
    ("robot/LF_finger_link_meshcol2", "bar_top_grip_collar_a"),
    ("robot/LF_finger_tip_link_visual", "bar_top_grip_collar_a"),
    ("robot/display_module_link_meshcol", "bar_top_grip_collar_a"),
    ("robot/display_module_link_visual", "bar_top_grip_collar_b"),
    ("robot/LF_finger_grip_insert_link_visual_extra", "bar_top_grip_collar_a"),
    ("robot/LF_finger_grip_insert_link_visual", "bar_top_grip_collar_a_extra"),
    ("robot/unknown", "bar_top_grip_collar_a"),
    ("robot/LF_palm_grip_insert_link_meshcol", "unknown"),
])
def test_non_exempt_pairs_fail_closed_and_actual_depth_wins(robot, pitcher):
    module = _load("m4_pair_clearance")
    tracker = module.PairClearance([0], [1], {0: robot, 1: pitcher})
    assert not tracker.excluded
    model = SimpleNamespace(geom_rbound=np.zeros(2))
    data = SimpleNamespace(geom_xpos=np.zeros((2, 3)), ncon=0, contact=[])
    tracker.observe(model, data, 1.0, lambda a, b: 0.004, 0.1)
    assert tracker.report()["excluding_intentional_grasp_pairs"]["minimum_m"] < 0.005
    data.ncon = 2
    data.contact = [SimpleNamespace(geom1=1, geom2=0, dist=0.001),
                    SimpleNamespace(geom1=0, geom2=1, dist=-0.0004)]
    def unexpected(*args):
        raise AssertionError("actual contact must never be replaced by confirmed distance/force")
    tracker.observe(model, data, 2.0, unexpected, 0.1, contact_force=unexpected)
    report = tracker.report()
    assert report["non_exempt_actual_contact"]["minimum_m"] == -0.0004
    assert report["excluding_intentional_grasp_pairs"]["minimum_m"] == -0.0004
    assert report["unrestricted"]["minimum_m"] == -0.0004
    assert all(row["contact_count"] == 0 for row in report["physical_collision_contacts"])


def test_collision_contacts_keep_their_own_depths_and_forces_not_visual_aliases():
    module = _load("m4_pair_clearance")
    pairs = module.EXEMPT_GEOM_PAIRS
    assert len(pairs) == len(set(pairs)) == 4
    ordered = list(dict.fromkeys(name for pair in pairs for name in pair))
    names = dict(enumerate(ordered))
    ids = {name: index for index, name in names.items()}
    tracker = module.PairClearance([ids[pair[0]] for pair in pairs],
                                   [ids[pairs[0][1]], ids[pairs[2][1]]], names)
    model = SimpleNamespace(geom_rbound=np.zeros(len(names)))
    data = SimpleNamespace(geom_xpos=np.zeros((len(names), 3)), ncon=4, contact=[
        SimpleNamespace(geom1=ids[b], geom2=ids[a], dist=-0.001 * (i + 1))
        for i, (a, b) in enumerate(pairs)
    ])
    measured = []
    def force(index):
        measured.append(index)
        return 2.0 + index
    tracker.observe(model, data, 1.0, lambda a, b: 0.02, 0.1, contact_force=force)
    assert measured == [0, 2], "visual aliases must not supply physical-contact forces"
    report = tracker.report()
    assert report["excluded_pairs"] == [list(pair) for pair in pairs]
    assert report["unrestricted"]["minimum_m"] == -0.004
    physical = report["physical_collision_contacts"]
    assert [row["minimum_m"] for row in physical] == [-0.001, -0.003]
    assert [row["maximum_normal_force_n"] for row in physical] == [2.0, 4.0]
    assert all(row["contact_count"] == row["force_samples"] == 1 for row in physical)
    # Repeat observations retain extrema, not the last reading.
    for contact in data.contact:
        contact.dist = -0.0001
    tracker.observe(model, data, 2.0, lambda a, b: 0.02, 0.1, contact_force=lambda i: 1.0)
    assert [row["minimum_m"] for row in physical] == [-0.001, -0.003]
    assert [row["maximum_normal_force_n"] for row in physical] == [2.0, 4.0]
    assert all(row["contact_count"] == row["force_samples"] == 2 for row in physical)


def test_ambiguous_names_cannot_expand_the_exact_mapping():
    module = _load("m4_pair_clearance")
    with pytest.raises(ValueError, match="unique"):
        module.PairClearance([0, 1], [2], {
            0: "robot/LF_finger_grip_insert_link_visual",
            1: "robot/LF_finger_grip_insert_link_visual", 2: "bar_top_grip_collar_a",
        })
    with pytest.raises(KeyError):
        module.PairClearance([0], [1], {0: "robot/unknown"})


def test_normal_contact_force_reads_the_actual_mujoco_wrench():
    run_m4 = _load("run_m4")
    def contact_force(model, data, index, wrench):
        assert (model, data, index) == ("model", "data", 7)
        wrench[:] = [1.25, 2, 3, 4, 5, 6]
    mujoco = SimpleNamespace(mj_contactForce=contact_force)
    assert run_m4._normal_contact_force(mujoco, "model", "data", 7) == 1.25


def test_pair_clearance_far_pairs_are_conservative_and_head_is_not_exempt():
    module = _load("m4_pair_clearance")
    names = {index: name for index, name in enumerate(
        [name for pair in module.INTENTIONAL_GRASP_PAIRS for name in pair]
        + ["robot/display_module_link_meshcol", "bar_spout_tip"])}
    tracker = module.PairClearance([4], [5], names)
    model = SimpleNamespace(geom_rbound=np.zeros(6))
    positions = np.zeros((6, 3))
    positions[5, 0] = 1.0
    data = SimpleNamespace(geom_xpos=positions, ncon=0, contact=[])
    def unexpected(*args):
        raise AssertionError("far pair should use its bounding-sphere lower bound")
    tracker.observe(model, data, 1.0, unexpected, 0.1)
    assert tracker.report()["excluding_intentional_grasp_pairs"]["minimum_m"] == 1.0


def test_m4_trajectory_runs_at_1khz_with_50hz_targets():
    config = json.loads(TRAJECTORY.read_text("utf-8"))
    assert config["physics_dt_s"] == 0.001
    assert config["control_dt_s"] == 0.02
    decimation = round(config["control_dt_s"] / config["physics_dt_s"])
    assert decimation == 20
    assert abs(decimation * config["physics_dt_s"] - config["control_dt_s"]) < 1e-12


def test_run_m4_parses_a_spawn_offset():
    run_m4 = _load("run_m4")
    args = run_m4._parser().parse_args(["--out", "x", "--spawn-offset", "0.002", "0", "-2"])
    assert tuple(args.spawn_offset) == (0.002, 0.0, -2.0)
    assert tuple(run_m4._parser().parse_args(["--out", "x"]).spawn_offset) == (0.0, 0.0, 0.0)


def test_segment_fraction_is_smooth_progress_through_the_keyframe():
    run_m4 = _load("run_m4")
    config = {"keyframes": [{"end_s": 1.0}, {"end_s": 3.0}]}
    assert run_m4._segment_fraction(config, 1.0) == 1.0
    assert run_m4._segment_fraction(config, 2.0) == pytest.approx(0.5)
    assert run_m4._segment_fraction(config, 9.0) == 1.0


# --------------------------------------------------------------------------- sweep tool
def test_sweep_grid_is_the_acceptance_grid():
    sweep = _load("m4_sweep")
    grid = sweep.default_grid()
    yaws = sorted(v[2] for v in grid.values() if v[:2] == (0.0, 0.0))
    assert yaws == [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    xy = sorted(v[:2] for v in grid.values() if v[:2] != (0.0, 0.0))
    assert xy == [(-0.002, 0.0), (0.0, -0.002), (0.0, 0.002), (0.002, 0.0)]
    assert all(v[2] == 0.0 for v in grid.values() if v[:2] != (0.0, 0.0))
    assert len(grid) == 13


def _summary(passes: bool) -> dict:
    whole = {"phase": "whole_run", "peak_applied_abs_nm": 1.1, "maximum_thermal_theta": 0.5,
             "above_continuous_max_contiguous_s": 0.0}
    return {
        "passes": passes,
        "payload": {"passes": True},
        "torque": {"passes": True, "per_joint_and_phase": {"LF_J0:whole_run": whole}},
        "determinism": {"qpos_sha256": "ab" * 32},
        "motion": {
            "checks": {"carried_grasp_translation": passes, "stable_final_set_down": True},
            "maximum_carried_grasp_translation_m": 0.0071,
            "maximum_carried_grasp_rotation_deg": 1.0,
            "maximum_final_position_error_m": 0.012,
            "spout_height_min_m": 0.03, "spout_height_max_m": 0.04, "pour_duration_s": 18.0,
            "minimum_pitcher_brewer_clearance_all_phases_m": 0.004,
            "minimum_robot_brewer_clearance_all_phases_m": 0.02,
            "pitcher_robot_pair_clearance": {
                "unrestricted": {"minimum_m": -0.001, "nearest_pair": ["a", "b"]},
                "excluding_intentional_grasp_pairs": {
                    "minimum_m": 0.002, "nearest_pair": ["c", "d"],
                },
            },
            "pitcher_robot_clearance": {"by_scope": {
                "carried": {"minimum_m": 0.009, "nearest_pair": "a;b", "time_s": 30.0},
                "spiral": {"minimum_m": 0.011, "nearest_pair": "a;b", "time_s": 31.0},
            }},
        },
        "stream": {
            "checks": {"inward_excursion": True},
            "maximum_inward_excursion_m": 0.005, "turn_count": 3.6, "first_radius_m": 0.004,
            "minimum_edge_margin_m": 0.008, "landing_success_rate": 1.0,
        },
    }


def test_sweep_point_row_reports_every_gate_and_the_margins(tmp_path):
    sweep = _load("m4_sweep")
    good = sweep.point_row("yaw-2.0deg", (0.0, 0.0, -2.0), 0, _summary(True))
    bad = sweep.point_row("dx+2mm", (0.002, 0.0, 0.0), 1, _summary(False))
    crash = sweep.point_row("dy-2mm", (0.0, -0.002, 0.0), 1, None)
    assert good["passes"] and good["failing_gates"] == ""
    assert good["carried_grasp_translation_mm"] == pytest.approx(7.1)
    assert good["set_down_error_mm"] == pytest.approx(12.0)
    assert good["min_pitcher_robot_clearance_carried_mm"] == pytest.approx(9.0)
    assert good["spiral_unrestricted_clearance_mm"] == pytest.approx(-1.0)
    assert good["spiral_excluding_intentional_grasp_pairs_clearance_mm"] == pytest.approx(2.0)
    assert good["motion.carried_grasp_translation"] is True and good["torque"] is True
    assert not bad["passes"] and bad["failing_gates"] == "motion.carried_grasp_translation"
    assert bad["dx_mm"] == pytest.approx(2.0)
    assert crash["passes"] is False and "crash" in crash["failing_gates"]
    table = tmp_path / "t.csv"
    sweep.write_table([good, bad, crash], table)
    with table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["point"] for row in rows] == ["yaw-2.0deg", "dx+2mm", "dy-2mm"]
    assert "stream.inward_excursion" in rows[0]


# --------------------------------------------------------------------------- torque log
def test_streamed_torque_log_matches_the_in_memory_summary_and_csv(tmp_path):
    run_m4 = _load("run_m4")
    rng = np.random.default_rng(3)
    rows = []
    for step, phase in enumerate(["lift"] * 300 + ["spiral"] * 400 + ["lift"] * 200):
        for joint in ("LF_J0_joint", "LF_J4_joint"):
            tau = float(rng.uniform(-2.0, 2.0))
            rows.append({
                "run_id": "r", "phase": phase, "joint": joint, "time_s": step * 0.001,
                "qd_pre_rad_s": float(rng.uniform(-3, 3)), "tau_raw_nm": tau * 1.1,
                "tau_applied_nm": tau, "curve_margin_nm": float(rng.uniform(-0.1, 1)),
                "effective_margin_nm": float(rng.uniform(-0.1, 1)),
                "thermal_theta_pre": float(rng.random()),
                "thermal_theta_post": float(rng.random()),
                "above_continuous": abs(tau) > 1.2, "derating_active": bool(rng.random() > 0.9),
                "curve_violation": bool(rng.random() > 0.95),
                "thermal_recompute_error": float(rng.random() * 1e-6),
            })
    log = run_m4._TorqueLog(tmp_path / "torque.csv", 0.001)
    for row in rows:
        log.add(row)
    log.close()
    expected = run_m4._summarize_torque(rows, 0.001)
    assert log.summary() == expected and list(log.summary()) == list(expected)
    assert log.count == len(rows)
    reference = tmp_path / "reference.csv"
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert (tmp_path / "torque.csv").read_bytes() == reference.read_bytes()


def test_key_frame_times_follow_the_phases():
    run_m4 = _load("run_m4")
    config = json.loads(TRAJECTORY.read_text("utf-8"))
    frames = tuple(Path(name) for name in ("start", "mid", "outer", "set_down"))
    times = {frame.name: time for time, frame in run_m4._key_frame_times(config, frames)}
    ends = {k["phase"]: k["end_s"] for k in config["keyframes"]}
    assert times["start"] == 0.5
    assert times["outer"] == pytest.approx(ends["spiral"] - 3.0)
    assert times["outer"] > times["mid"] > ends["pour_in"]
    assert ends["retreat"] < times["set_down"] < ends["final_settle"]
