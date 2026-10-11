"""The A4 motion chain (barista/chain/): fast, simulation-free checks.

The physical evidence (the continuous run, its determinism, the M4 gates on the walked
stance, the perturbation sweep, the torque audit) comes from ``barista/chain/run_chain.py``
and ``barista/chain/pour.py``; these tests pin the pieces those runs rely on.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / "barista" / "chain"


def _load(stem: str):
    name = f"barista_chain_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CHAIN / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load("common")
beats = _load("beats")
footprint = _load("footprint")
pour = _load("pour")
approach = _load("approach")


def _valid_log() -> list[dict]:
    spans = [("dance", 0, 12), ("chime", 11, 12), ("stop", 12, 15), ("excited_jump", 15, 18),
             ("exclaim", 15.7, 17.2), ("look", 18, 24.5), ("listen", 18, 22), ("confirm", 22, 24.5),
             ("turn", 24.5, 29), ("walk", 29, 38), ("align", 38, 62), ("pour", 62, 105.5),
             ("set_down", 105.5, 119)]
    return [{"beat": b, "t_start_s": float(a), "t_end_s": float(e)} for b, a, e in spans]


def test_beat_vocabulary_is_the_supervisors() -> None:
    assert common.BEAT_NAMES == ("dance", "chime", "stop", "excited_jump", "exclaim", "look",
                                 "listen", "confirm", "turn", "walk", "align", "pour",
                                 "set_down", "bow")
    assert common.BEATS_SCHEMA == "jumper-barista-beats/1"
    assert beats.DEFAULT_HOLDS_S == {"exclaim": 1.5, "listen": 4.0, "confirm": 2.5}


def test_a_valid_beat_log_validates_and_serialises() -> None:
    log = _valid_log()
    assert beats.validate(log) == []
    doc = beats.document(log, commit="abc", physics_dt_s=0.001, control_dt_s=0.005)
    assert doc["schema"] == "jumper-barista-beats/1"
    assert set(doc["beats"][0]) == {"beat", "t_start_s", "t_end_s"}
    json.dumps(doc)


@pytest.mark.parametrize("mutate, message", [
    (lambda log: log.__setitem__(3, {**log[3], "t_start_s": 14.0}), "not contiguous"),
    (lambda log: log.append({"beat": "moonwalk", "t_start_s": 1.0, "t_end_s": 2.0}), "unknown beat"),
    (lambda log: log.__setitem__(4, {**log[4], "t_end_s": 19.0}), "not inside excited_jump"),
    (lambda log: log.__setitem__(1, {**log[1], "t_start_s": 12.5, "t_end_s": 13.0}),
     "not inside dance"),
])
def test_beat_log_problems_are_caught(mutate, message) -> None:
    log = _valid_log()
    mutate(log)
    assert any(message in p for p in beats.validate(log))


def test_footprint_rejects_states_from_another_scene() -> None:
    """The spike's p8 crash: plain-scene width 29 replayed in a bar scene of width 36."""
    with pytest.raises(footprint.StateWidthError, match="width 29 .* nq=36"):
        footprint.check_state_width(np.zeros((5, 29)), 36, "cafe_bar")
    ok = footprint.check_state_width(np.zeros((5, 36)), 36, "cafe_bar")
    assert ok.shape == (5, 36)


def test_footprint_envelope_on_a_small_model() -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco><worldbody><body name="b"><freejoint/>
      <geom name="robot/g" type="sphere" size="0.05" pos="0.1 0 0"/>
      <geom name="prop" type="sphere" size="0.2" pos="0 0 0" contype="0" conaffinity="0"/>
      </body><body name="p"><freejoint/><geom name="pitcher" type="box" size=".01 .01 .01"/></body>
      </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    states = np.tile(model.qpos0, (3, 1))
    states[1, 0] = 0.2  # body moved +x 0.2 m: robot geom at x 0.3
    r, i = footprint.envelope_radius(model, states, (0.0, 0.0), "toy")
    assert i == 1
    assert r == pytest.approx(0.3 + 0.05)
    with pytest.raises(footprint.StateWidthError):
        footprint.envelope_radius(model, states[:, :7], (0.0, 0.0), "toy")


def test_cafe_trajectory_moves_only_world_positions_and_rates() -> None:
    accepted = json.loads(common.M4_TRAJECTORY.read_text())
    moved = common.cafe_m4_trajectory(0.001, 20)
    offset = np.asarray(common.cafe_dims()["brew_station"]["compiled_offset_m"])
    assert moved["keyframes"] == accepted["keyframes"]
    assert moved["thresholds"] == accepted["thresholds"]
    assert moved["support_pour_rad"] == accepted["support_pour_rad"]
    for key in ("base_start",):
        np.testing.assert_allclose(moved[key]["position_m"],
                                   np.asarray(accepted[key]["position_m"]) + offset, atol=1e-9)
    np.testing.assert_allclose(moved["set_down_target_m"],
                               np.asarray(accepted["set_down_target_m"]) + offset, atol=1e-9)
    assert moved["physics_dt_s"] == 0.001
    assert moved["control_dt_s"] == pytest.approx(0.02)
    assert moved["stream_dt_s"] == pytest.approx(0.02)
    # pour_start is the transferred M4 start (A1's contract)
    pose = common.cafe_dims()["route"]["pour_start_pose"]
    assert moved["base_start"]["position_m"][0] == pytest.approx(pose["x_m"], abs=1e-6)
    assert moved["base_start"]["position_m"][1] == pytest.approx(pose["y_m"], abs=1e-6)


def test_slim_csv_keeps_every_100th_torque_substep_and_all_other_rows() -> None:
    slim = pour._SlimCsv(csv)
    handle = io.StringIO()
    rows = [{"substep": s, "joint": j, "tau_applied_nm": 0.1} for s in range(250) for j in "ab"]
    writer = slim.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    kept = list(csv.DictReader(io.StringIO(handle.getvalue())))
    assert [int(r["substep"]) for r in kept] == [0, 0, 100, 100, 200, 200]
    handle = io.StringIO()
    writer = slim.DictWriter(handle, fieldnames=["time_s", "phase"])
    writer.writeheader()
    writer.writerows([{"time_s": t, "phase": "x"} for t in range(7)])
    assert len(list(csv.DictReader(io.StringIO(handle.getvalue())))) == 7


def test_perturbed_state_is_a_rigid_planar_move() -> None:
    qpos = np.zeros(36)
    qpos[:7] = [0.8, 0.0, 0.1, math.cos(0.05), 0.0, 0.0, math.sin(0.05)]
    qvel = np.zeros(35)
    qvel[0] = 0.01
    state = {"qpos": qpos, "qvel": qvel}
    out = pour.perturbed_state(state, 0.002, -0.002, 2.0)
    assert out["qpos"][0] == pytest.approx(0.802)
    assert out["qpos"][1] == pytest.approx(-0.002)
    assert math.degrees(common.yaw_of(out["qpos"][3:7])) == pytest.approx(math.degrees(0.1) + 2.0)
    assert np.linalg.norm(out["qvel"][:3]) == pytest.approx(0.01)
    np.testing.assert_array_equal(out["qpos"][7:], qpos[7:])


def test_yaw_sweep_grid_targets_absolute_arrival_yaw_and_finds_bands() -> None:
    sweep = _load("pour_sweep")
    grid = sweep.yaw_grid(-1.0, 3.0, 0.5)
    assert grid == [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    with pytest.raises(ValueError):
        sweep.yaw_grid(-1.0, 3.0, 0.3)
    qpos = np.zeros(36)
    qpos[3:7] = common.yaw_quat(math.radians(-0.66))
    measured = sweep.arrival_yaw_error_deg(qpos, 0.0)
    assert measured == pytest.approx(-0.66)
    cases = sweep.yaw_cases(grid, measured)
    for value, (dx, dy, dyaw) in zip(grid, cases.values(), strict=True):
        assert dx == 0.0 and dy == 0.0
        moved = pour.perturbed_state({"qpos": qpos, "qvel": np.zeros(35)}, dx, dy, dyaw)
        assert sweep.arrival_yaw_error_deg(moved["qpos"], 0.0) == pytest.approx(value)
    found = sweep.bands([(1.0, True), (-1.0, False), (0.0, True), (0.5, True), (1.5, False),
                         (2.0, True)])
    assert [(b["from_deg"], b["to_deg"], b["width_deg"]) for b in found] == [
        (0.0, 1.0, 1.0), (2.0, 2.0, 0.0)]
    assert found[0]["centre_deg"] == 0.5


def test_rotation_log_round_trips() -> None:
    axis = np.array([0.3, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    angle = 0.7
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    np.testing.assert_allclose(common.rot_log(rot), axis * angle, atol=1e-12)
    np.testing.assert_allclose(common.rot_log(np.eye(3)), np.zeros(3), atol=1e-15)


def test_approach_pose_interpolation_takes_the_short_way_round() -> None:
    a = (0.72, 0.0, 0.1, math.radians(179.0))
    b = (0.80, 0.0, 0.1, math.radians(-179.0))
    mid = approach.interp_pose(a, b, 0.5)
    assert mid[0] == pytest.approx(0.76)
    assert abs(common.wrap(mid[3] - math.pi)) < 1e-9


def test_approach_sequence_lifts_lf_only_after_the_five_steps() -> None:
    names = [s for s, _ in approach.ApproachPlan.stages(object.__new__(approach.ApproachPlan))]
    assert names.index("lf_up") > max(names.index(f"step_{leg}") for leg in approach.STEP_ORDER)
    assert names.index("shift_2") > names.index("lf_high")
    assert names[-2:] == ["lf_place", "hold"]
    assert approach.KP == 10.0 and approach.KD == 0.5  # run_m4's own servo gains


def test_pour_phases_partition_the_m4_timeline() -> None:
    accepted = json.loads(common.M4_TRAJECTORY.read_text())
    phases = {k["phase"] for k in accepted["keyframes"]}
    assert phases == set(pour.PALM_PHASES) | set(pour.SPOUT_PHASES)
    assert not set(pour.PALM_PHASES) & set(pour.SPOUT_PHASES)
    run_chain = _load("run_chain")
    assert set(run_chain.POUR_PHASES) | set(run_chain.SET_DOWN_PHASES) == phases
