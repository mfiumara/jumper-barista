"""Closed-loop, spout-space re-targeting of the accepted M4 pour (supervisor decision on A4).

Simulation only. The accepted verifier ``barista/verification/run_m4.py`` runs unmodified:
every M4 gate (turns, landings, stream obstruction, spout height, set-down, brewer and
free-leg contacts, payload, torque audit) is its own. What changes is only the *input*:

* the scene is ``cafe_bar`` with the accepted trajectory rigidly moved into it (A1's
  transfer) and re-rated to 1 kHz physics / 50 Hz targets (MOTION_CHAIN.md Q2, R3);
* in ``retarget`` mode, the four LF arm targets that ``run_m4._target`` returns are
  corrected every 20 ms by damped least-squares IK so the pour is expressed in the brew
  station's frame instead of the robot's: before the grasp and after the set-down the
  palm frame tracks the palm pose the accepted keyframes command in the nominal replay;
  while the pitcher is carried, the *measured* spout outlet (its offset in the measured
  palm frame, i.e. the grasp as it actually is) tracks the nominal commanded spout
  position. The reference comes from one nominal replay recorded in ``reference`` mode.
  With the robot exactly where the nominal replay had it, the IK returns the accepted
  keyframes unchanged (residual 0), so this is a correction, not a new trajectory;
* in ``attached`` use (``barista/chain/run_chain.py``), run_m4 runs inside the chain's
  already-running world: its env construction, reset and the two initial state writes are
  skipped, so the pour starts from the walked arrival state with no teleport.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("barista_chain_common", Path(__file__).with_name("common.py"))
    common = _ilu.module_from_spec(_spec)
    sys.modules.setdefault("barista_chain_common", common)
    _spec.loader.exec_module(common)
else:  # pragma: no cover - loaded by path everywhere in this repo
    from . import common

ARM = ("LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint")
#: Phases in which the palm (the grasp frame) is servoed: the pitcher is on the counter.
PALM_PHASES = ("settle", "approach", "close", "lower", "set_down", "release",
               "clear_retreat", "retreat", "final_settle")
#: Phases in which the measured spout outlet is servoed: the pitcher is in the claw.
SPOUT_PHASES = ("lift", "lift_settle", "lift_hold", "transit", "pour_in", "spiral",
                "pour_out", "return")
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT_M_PER_RAD = 0.05
REGULARISATION_M_PER_RAD = 0.002
IK_ITERATIONS = 8
#: Set-down: xy integral gain (1/s) and clamp on the measured pitcher origin during ``lower``.
LOWER_KI = 2.0
LOWER_I_CLAMP_M = 0.015


class ArmIK:
    """Kinematics of the LF arm on a scratch MjData (never the simulated one)."""

    def __init__(self, model) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)

        def one(kind, count, suffix):
            hits = [i for i in range(getattr(model, count))
                    if (mujoco.mj_id2name(model, kind, i) or "").endswith(suffix)]
            if len(hits) != 1:
                raise RuntimeError(f"expected one {suffix}, found {hits}")
            return hits[0]

        self.palm = one(mujoco.mjtObj.mjOBJ_BODY, "nbody", "LF_palm_link")
        self.base = one(mujoco.mjtObj.mjOBJ_BODY, "nbody", "base_link")
        self.spout = one(mujoco.mjtObj.mjOBJ_SITE, "nsite", "pitcher_spout_tip")
        joints = [one(mujoco.mjtObj.mjOBJ_JOINT, "njnt", f"/{n}") for n in ARM]
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.vadr = np.array([model.jnt_dofadr[j] for j in joints])
        self.lo = np.array([model.jnt_range[j, 0] for j in joints])
        self.hi = np.array([model.jnt_range[j, 1] for j in joints])

    def set(self, qpos: np.ndarray, arm: np.ndarray | None = None) -> None:
        self.data.qpos[:] = qpos
        if arm is not None:
            self.data.qpos[self.qadr] = arm
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_comPos(self.model, self.data)

    def palm_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.data.xpos[self.palm].copy(),
                self.data.xmat[self.palm].reshape(3, 3).copy())

    def spout_world(self) -> np.ndarray:
        return self.data.site_xpos[self.spout].copy()

    def _jac(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.palm)
        return jacp[:, self.vadr], jacr[:, self.vadr]

    def solve(self, qpos: np.ndarray, arm0: np.ndarray, *, palm_pos=None, palm_rot=None,
              spout_target=None, spout_in_palm=None) -> tuple[np.ndarray, dict]:
        """Damped least squares from ``arm0`` (the accepted keyframe target)."""
        arm = np.asarray(arm0, dtype=float).copy()
        for _ in range(IK_ITERATIONS):
            self.set(qpos, arm)
            p, rot = self.palm_pose()
            rows, errs = [], []
            if spout_target is not None:
                point = p + rot @ spout_in_palm
                jp, _ = self._jac(point)
                rows.append(POSITION_WEIGHT * jp)
                errs.append(POSITION_WEIGHT * (spout_target - point))
            if palm_pos is not None:
                jp, _ = self._jac(p)
                rows.append(POSITION_WEIGHT * jp)
                errs.append(POSITION_WEIGHT * (palm_pos - p))
            if palm_rot is not None:
                _, jr = self._jac(p)
                rows.append(ORIENTATION_WEIGHT_M_PER_RAD * jr)
                errs.append(ORIENTATION_WEIGHT_M_PER_RAD * common.rot_log(palm_rot @ rot.T))
            rows.append(REGULARISATION_M_PER_RAD * np.eye(len(arm)))
            errs.append(REGULARISATION_M_PER_RAD * (np.asarray(arm0) - arm))
            jac = np.vstack(rows)
            err = np.concatenate(errs)
            step, *_ = np.linalg.lstsq(jac, err, rcond=None)
            arm = np.clip(arm + step, self.lo, self.hi)
        self.set(qpos, arm)
        p, rot = self.palm_pose()
        info = {}
        if spout_target is not None:
            info["spout_residual_m"] = float(np.linalg.norm(spout_target - (p + rot @ spout_in_palm)))
        if palm_pos is not None:
            info["palm_residual_m"] = float(np.linalg.norm(palm_pos - p))
        if palm_rot is not None:
            info["palm_rot_residual_rad"] = float(np.linalg.norm(common.rot_log(palm_rot @ rot.T)))
        return arm, info


class _Hooks:
    """run_m4 hooks: record the nominal reference, or re-target against it."""

    def __init__(self, run_m4, mode: str, reference: dict | None, control_dt: float) -> None:
        self.run_m4 = run_m4
        self.mode = mode
        self.reference = reference
        self.control_dt = control_dt
        self.orig_target = run_m4._target
        self.orig_blend = run_m4._blend_from
        self.env = None
        self.ik = None
        self.rows: list[dict] = []
        self._tick_cache: dict[int, tuple] = {}
        self._integral = np.zeros(3)
        self._pitcher = None

    def _pitcher_body(self) -> int:
        if self._pitcher is None:
            import mujoco

            model = self.env.sim.mj_model
            self._pitcher = next(b for b in range(model.nbody)
                                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                                 .endswith("pitcher"))
        return self._pitcher

    # -------------------------------------------------------------- helpers
    def _tick(self, time_s: float) -> int | None:
        tick = round(time_s / self.control_dt)
        return tick if abs(time_s - tick * self.control_dt) < 1e-9 else None

    def _data(self):
        return self.env.sim._datas[0]

    def _measured(self):
        data = self._data()
        ik = self.ik
        qpos = data.qpos.copy()
        palm_pos = data.xpos[ik.palm].copy()
        palm_rot = data.xmat[ik.palm].reshape(3, 3).copy()
        spout = data.site_xpos[ik.spout].copy()
        base_pos = data.xpos[ik.base].copy()
        base_quat = data.xquat[ik.base].copy()
        return qpos, palm_pos, palm_rot, spout, base_pos, base_quat

    # -------------------------------------------------------------- patched run_m4 API
    def target(self, config, time_s):
        phase, arm, finger = self.orig_target(config, time_s)
        tick = self._tick(time_s)
        if tick is None or self.env is None:
            return phase, arm, finger
        if self.ik is None:
            self.ik = ArmIK(self.env.sim.mj_model)
        if tick in self._tick_cache:
            return self._tick_cache[tick]
        qpos, palm_pos, palm_rot, spout, base_pos, base_quat = self._measured()
        spout_in_palm = palm_rot.T @ (spout - palm_pos)
        row = {"tick": tick, "time_s": time_s, "phase": phase, "arm_ref": np.asarray(arm, float),
               "finger": float(finger), "base_pos": base_pos, "base_quat": base_quat,
               "palm_meas": palm_pos, "spout_meas": spout, "spout_in_palm": spout_in_palm}
        out_arm = np.asarray(arm, dtype=float)
        if self.mode == "reference":
            self.ik.set(qpos, out_arm)
            cmd_pos, cmd_rot = self.ik.palm_pose()
            row.update(palm_cmd=cmd_pos, palm_cmd_rot=cmd_rot.ravel(),
                       spout_cmd=cmd_pos + cmd_rot @ spout_in_palm)
        elif self.mode == "retarget":
            ref = self.reference
            index = int(np.searchsorted(ref["tick"], tick))
            if index >= len(ref["tick"]) or int(ref["tick"][index]) != tick:
                raise RuntimeError(f"reference has no control tick {tick}")
            palm_cmd = ref["palm_cmd"][index]
            palm_cmd_rot = ref["palm_cmd_rot"][index].reshape(3, 3)
            bias = np.zeros(3)
            if "setdown_bias_xy" in ref:
                bias[:2] = np.asarray(ref["setdown_bias_xy"], dtype=float)
            if phase == "lower" and "pitcher_cmd" in ref:
                # Set the pitcher down on its pickup spot: the accepted keyframes return it
                # ~20 mm short (the reference's own bias); servo the measured pitcher origin
                # onto the reference path minus that bias, with a small xy integral term.
                touch = float(ref["touchdown_time_s"])
                lower_start = float(ref["time_s"][int(np.argmax(ref["phase"] == "lower"))])
                ramp = common.smooth((time_s - lower_start) / max(touch - 0.2 - lower_start, 1e-6))
                pitcher_meas = self._data().xpos[self._pitcher_body()].copy()
                desired_meas = ref["pitcher_meas"][index] - bias * ramp
                self._integral[:2] = np.clip(
                    self._integral[:2] + LOWER_KI * self.control_dt
                    * (desired_meas[:2] - pitcher_meas[:2]), -LOWER_I_CLAMP_M, LOWER_I_CLAMP_M)
                target_point = ref["pitcher_cmd"][index] - bias * ramp + self._integral
                pitcher_in_palm = palm_rot.T @ (pitcher_meas - palm_pos)
                out_arm, info = self.ik.solve(
                    qpos, out_arm, palm_rot=palm_cmd_rot,
                    spout_target=target_point, spout_in_palm=pitcher_in_palm)
                row["pitcher_error_m"] = float(np.linalg.norm(desired_meas[:2] - pitcher_meas[:2]))
            elif phase in SPOUT_PHASES:
                out_arm, info = self.ik.solve(
                    qpos, out_arm, palm_rot=palm_cmd_rot,
                    spout_target=ref["spout_cmd"][index], spout_in_palm=spout_in_palm)
            else:
                after = phase in ("set_down", "release", "clear_retreat", "retreat", "final_settle")
                shift = (self._integral - bias) if after else np.zeros(3)
                out_arm, info = self.ik.solve(qpos, out_arm, palm_pos=palm_cmd + shift,
                                              palm_rot=palm_cmd_rot)
            row.update(info)
            row["arm_cmd"] = out_arm
            row["correction_rad"] = out_arm - np.asarray(arm, float)
        result = (phase, out_arm, finger)
        self._tick_cache = {tick: result}
        self.rows.append(row)
        return result

    def blend_from(self, config, time_s, start_arm):
        """Clear-retreat blend from the measured set-down arm toward the corrected keyframe."""
        nominal = self.orig_blend(config, time_s, start_arm)
        if self.mode != "retarget":
            return nominal
        _, corrected, _ = self.target(config, time_s)
        _, keyframe_arm, _ = self.orig_target(config, time_s)
        correction = np.asarray(corrected) - np.asarray(keyframe_arm)
        previous = 0.0
        fraction = 1.0
        for keyframe in config["keyframes"]:
            end = float(keyframe["end_s"])
            if time_s <= end:
                fraction = self.run_m4._smooth((time_s - previous) / max(end - previous, 1e-12))
                break
            previous = end
        return nominal + correction * fraction

    def reference_arrays(self) -> dict:
        keys = ("tick", "time_s", "arm_ref", "finger", "base_pos", "base_quat", "palm_meas",
                "spout_meas", "spout_in_palm", "palm_cmd", "palm_cmd_rot", "spout_cmd")
        out = {k: np.asarray([r[k] for r in self.rows]) for k in keys}
        out["phase"] = np.asarray([r["phase"] for r in self.rows])
        return out


TORQUE_CSV_STRIDE = 100


class _SlimCsv:
    """``csv`` stand-in for run_m4's file export only: at 1 kHz its per-substep torque CSV is
    ~390 MB per pour. With ``slim_torque_csv`` every row is still reduced by run_m4's own
    ``_summarize_torque`` (the gate); only the *file copy* keeps one substep in 100 (all 22
    joints of it). Motion and stream CSVs are written in full."""

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def DictWriter(self, handle, fieldnames, *a, **k):
        writer = self._real.DictWriter(handle, fieldnames, *a, **k)
        if "tau_applied_nm" not in fieldnames:
            return writer

        class _Writer:
            def writeheader(self_inner):
                return writer.writeheader()

            def writerows(self_inner, rows):
                return writer.writerows(
                    r for r in rows if int(r["substep"]) % TORQUE_CSV_STRIDE == 0)

        return _Writer()


def run_pour(out: Path, *, mode: str = "retarget", reference_path: Path | None = None,
             attach_env=None, start_state: dict | None = None, render: bool = False,
             save_render_states: bool = True, on_substep=None, spawn_offset=(0.0, 0.0, 0.0),
             slim_torque_csv: bool = True) -> dict:
    """Run the unmodified M4 verifier with the cafe transfer at 1 kHz and the chosen hook.

    ``attach_env``: an already-running jumper.barista env in the cafe (the chain). run_m4's
    env construction, reset, close and its two initial state writes are skipped.
    ``start_state``: dict(qpos, qvel, thermal) written instead of run_m4's spawn (sweep).
    ``on_substep``: called after every physics substep (the chain's logger).
    """
    import mjlab.envs
    import torch

    import scenes
    import tasks

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / common.PHYSICS_HZ
    decimation = common.M4_DECIMATION_1K
    run_m4 = common.load_run_m4()
    config = common.cafe_m4_trajectory(dt, decimation)
    if any(spawn_offset):
        # Standalone sweep only: spawn the M4 start pose rigidly off by (dx, dy, dyaw).
        # The station, bed centre and set-down target stay where they are.
        config["base_start"]["position_m"][0] += float(spawn_offset[0])
        config["base_start"]["position_m"][1] += float(spawn_offset[1])
        config["base_start"]["yaw_rad"] = float(config["base_start"]["yaw_rad"]) + math.radians(
            float(spawn_offset[2]))
        config["a4_chain"]["spawn_offset_m_m_deg"] = [float(v) for v in spawn_offset]
    trajectory = out / "m4_spiral_pour_cafe_1k.json"
    trajectory.write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    reference = None
    if mode == "retarget":
        loaded = np.load(reference_path)
        reference = {k: loaded[k] for k in loaded.files}
    hooks = _Hooks(run_m4, mode, reference, dt * decimation)

    saved = {
        "decimation": run_m4.CONTROL_DECIMATION, "target": run_m4._target,
        "blend": run_m4._blend_from, "env_cls": mjlab.envs.ManagerBasedRlEnv,
        "load_env_cfg": tasks.load_env_cfg, "apply": scenes.apply, "argv": sys.argv,
    }
    real_cls = saved["env_cls"]
    real_load = saved["load_env_cfg"]
    real_apply = saved["apply"]
    restore = []

    def load_env_cfg(*args, **kwargs):
        cfg = real_load(*args, **kwargs)
        cfg.sim.mujoco.timestep = dt
        return cfg

    def apply(env_cfg, scene_id, **kwargs):
        return real_apply(env_cfg, common.SCENE if scene_id == "coffee_bar" else scene_id, **kwargs)

    def make_env(*args, **kwargs):
        if attach_env is None:
            env = real_cls(*args, **kwargs)
        else:
            env = attach_env
            for name in ("reset", "close"):
                restore.append((env, name, env.__dict__.get(name)))
                setattr(env, name, lambda *a, **k: None)
        hooks.env = env
        robot = env.scene["robot"]
        if attach_env is not None or start_state is not None:
            _install_state_writes(robot, env, start_state, restore, torch)
        if on_substep is not None:
            sim = env.sim
            real_step = sim.step

            def step(*a, **k):
                result = real_step(*a, **k)
                on_substep()
                return result

            restore.append((sim, "step", sim.__dict__.get("step")))
            sim.step = step
        return env

    run_m4.CONTROL_DECIMATION = decimation
    run_m4._target = hooks.target
    run_m4._blend_from = hooks.blend_from
    real_csv = run_m4.csv
    if slim_torque_csv:
        run_m4.csv = _SlimCsv(real_csv)
    mjlab.envs.ManagerBasedRlEnv = make_env
    tasks.load_env_cfg = load_env_cfg
    scenes.apply = apply
    argv = ["run_m4.py", "--out", str(out), "--trajectory", str(trajectory)]
    if save_render_states:
        argv.append("--save-render-states")
    if not render:
        argv.append("--no-render")
    sys.argv = argv
    try:
        code = run_m4.main()
    finally:
        run_m4.CONTROL_DECIMATION = saved["decimation"]
        run_m4.csv = real_csv
        run_m4._target = saved["target"]
        run_m4._blend_from = saved["blend"]
        mjlab.envs.ManagerBasedRlEnv = real_cls
        tasks.load_env_cfg = real_load
        scenes.apply = real_apply
        sys.argv = saved["argv"]
        for obj, name, previous in reversed(restore):
            if previous is None:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, previous)
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    arrays = hooks.reference_arrays() if mode == "reference" else None
    if arrays is not None:
        np.savez_compressed(out / "pour_reference.npz", **arrays)
    retarget = _retarget_report(hooks.rows) if mode == "retarget" else None
    if retarget is not None:
        common_json = out / "retarget.json"
        common_json.write_text(json.dumps(retarget, indent=1) + "\n", "utf-8")
    return {"exit_code": code, "passes": bool(summary["passes"]), "summary": summary,
            "retarget": retarget, "out": str(out)}


def _install_state_writes(robot, env, start_state, restore, torch) -> None:
    """Skip (attached) or replace (sweep) run_m4's two initial teleport writes."""
    calls = {"pose": 0, "joint": 0}
    real_pose = robot.write_root_link_pose_to_sim
    real_joint = robot.write_joint_state_to_sim

    def pose(root_pose, *a, **k):
        calls["pose"] += 1
        if calls["pose"] > 1:
            return real_pose(root_pose, *a, **k)
        if start_state is None:
            return None  # attached: the chain's arrival pose stands, no teleport
        data = env.sim._datas[0]
        qpos = np.asarray(start_state["qpos"], dtype=float)
        root = torch.tensor([qpos[:7]], dtype=root_pose.dtype)
        real_pose(root, *a, **k)
        vel = torch.tensor([np.asarray(start_state["qvel"], dtype=float)[:6]], dtype=root_pose.dtype)
        robot.write_root_link_velocity_to_sim(vel)
        del data
        return None

    def joint(pos, vel, *a, **k):
        calls["joint"] += 1
        if calls["joint"] > 1:
            return real_joint(pos, vel, *a, **k)
        if start_state is None:
            return None
        names = [str(n) for n in start_state["joint_names"]]
        order = list(robot.joint_names)
        jp = torch.tensor([[float(start_state["joint_pos"][names.index(n)]) for n in order]],
                          dtype=pos.dtype)
        jv = torch.tensor([[float(start_state["joint_vel"][names.index(n)]) for n in order]],
                          dtype=pos.dtype)
        result = real_joint(jp, jv, *a, **k)
        servo = robot.actuators[0]
        if "thermal" in start_state and getattr(servo, "_thermal", None) is not None:
            servo._thermal[:] = torch.tensor(
                [[float(start_state["thermal"][names.index(n)]) for n in servo.target_names]],
                dtype=servo._thermal.dtype)
        return result

    for name, fn in (("write_root_link_pose_to_sim", pose), ("write_joint_state_to_sim", joint)):
        restore.append((robot, name, robot.__dict__.get(name)))
        setattr(robot, name, fn)


def _retarget_report(rows: list[dict]) -> dict:
    corr = np.asarray([np.abs(r["correction_rad"]).max() for r in rows])
    spout = [r["spout_residual_m"] for r in rows if "spout_residual_m" in r]
    palm = [r["palm_residual_m"] for r in rows if "palm_residual_m" in r]
    steps = np.asarray([r["arm_cmd"] for r in rows])
    jumps = np.abs(np.diff(steps, axis=0)).max(axis=1) if len(steps) > 1 else np.zeros(1)
    return {
        "ticks": len(rows),
        "max_abs_correction_rad": float(corr.max()),
        "mean_abs_correction_rad": float(corr.mean()),
        "max_spout_ik_residual_m": float(max(spout)) if spout else None,
        "max_palm_ik_residual_m": float(max(palm)) if palm else None,
        "max_target_step_per_tick_rad": float(jumps.max()),
        "first_base_pos_m": rows[0]["base_pos"].tolist() if rows else None,
        "first_base_yaw_deg": math.degrees(common.yaw_of(rows[0]["base_quat"])) if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("reference", "retarget", "open_loop"), default="retarget")
    parser.add_argument("--reference", type=Path, default=None,
                        help="pour_reference.npz from a --mode reference run")
    parser.add_argument("--start-state", type=Path, default=None,
                        help="arrival_state.npz from the chain; the pour starts there")
    parser.add_argument("--dx", type=float, default=0.0, help="arrival perturbation, m")
    parser.add_argument("--dy", type=float, default=0.0, help="arrival perturbation, m")
    parser.add_argument("--dyaw-deg", type=float, default=0.0, help="arrival perturbation, deg")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    common.headless()
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    use_backend(resolve(backend="native", device="cpu", num_envs=1, training=False))
    start = None
    spawn = (args.dx, args.dy, args.dyaw_deg)
    if args.start_state is not None:
        start = perturbed_state(np.load(args.start_state), args.dx, args.dy, args.dyaw_deg)
        spawn = (0.0, 0.0, 0.0)
    result = run_pour(args.out, mode=args.mode, reference_path=args.reference,
                      start_state=start, render=args.render, spawn_offset=spawn)
    record = {"mode": args.mode, "exit_code": result["exit_code"], "passes": result["passes"],
              "perturbation": {"dx_m": args.dx, "dy_m": args.dy, "dyaw_deg": args.dyaw_deg},
              "start_state": str(args.start_state) if args.start_state else None,
              "retarget": result["retarget"],
              "failing_gates": failing_gates(result["summary"])}
    (Path(args.out) / "pour_result.json").write_text(json.dumps(record, indent=1) + "\n", "utf-8")
    print(json.dumps(record, indent=1))
    return 0 if result["passes"] else 1


def failing_gates(summary: dict) -> list[str]:
    failed = [f"motion.{k}" for k, v in summary["motion"]["checks"].items() if not v]
    failed += [f"stream.{k}" for k, v in summary["stream"]["checks"].items() if not v]
    if not summary["torque"]["passes"]:
        failed.append("torque")
    if not summary["payload"]["passes"]:
        failed.append("payload")
    return failed


def perturbed_state(state, dx: float, dy: float, dyaw_deg: float) -> dict:
    """Rigidly move a saved robot state by (dx, dy, dyaw) about its own base (the sweep)."""
    out = {k: np.asarray(state[k]) for k in state.files} if hasattr(state, "files") else dict(state)
    qpos = np.asarray(out["qpos"], dtype=float).copy()
    qvel = np.asarray(out["qvel"], dtype=float).copy()
    dyaw = math.radians(dyaw_deg)
    c, s = math.cos(dyaw), math.sin(dyaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    qpos[0] += dx
    qpos[1] += dy
    w1, x1, y1, z1 = common.yaw_quat(dyaw)
    w2, x2, y2, z2 = qpos[3:7]
    qpos[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                 w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    qvel[0:3] = rot @ qvel[0:3]
    out["qpos"] = qpos
    out["qvel"] = qvel
    return out


if __name__ == "__main__":
    raise SystemExit(main())
