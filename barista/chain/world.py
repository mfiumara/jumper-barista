"""The chain's one continuous world: jumper.barista in ``cafe_bar`` at 1 kHz physics, driven by
the jumper app's own controller (``deploy/fsm`` via ``rl/mjrl/app_play.py::AppPlayer``) with a
scripted pad, plus direct joint-target segments, all through the M4 audited servo.

Adapted from the accepted A4 spike harness (MOTION_CHAIN.md, "Harness"). Every physics
substep is audited; the whole run is reduced exactly as run_m4's torque gate reduces it.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

common = sys.modules.get("barista_chain_common")
if common is None:  # loaded by path
    _spec = importlib.util.spec_from_file_location(
        "barista_chain_common", Path(__file__).with_name("common.py"))
    common = importlib.util.module_from_spec(_spec)
    sys.modules["barista_chain_common"] = common
    _spec.loader.exec_module(common)

FSM_EXTENSION = common.REPO / "deploy" / "fsm" / "target" / "release" / "libmjrl_fsm.dylib"
RENDER_STRIDE = 20  # substeps per render-rate state at 1 kHz: 50 fps


def load_fsm_extension(path: Path = FSM_EXTENSION):
    if "mjrl_fsm" in sys.modules:
        return sys.modules["mjrl_fsm"]
    if not path.exists():
        raise SystemExit(f"{path} is missing: build the bundle first (scripts/deploy.py)")
    loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(path))
    spec = importlib.util.spec_from_file_location("mjrl_fsm", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["mjrl_fsm"] = module
    return module


@dataclass
class PadState:
    buttons: set = field(default_factory=set)
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0


class ScriptedPad:
    """Stands in for ``controller.open()``: what an operator's pad would report each tick."""

    connected = True

    def __init__(self) -> None:
        self.s = PadState()

    def state(self) -> PadState:
        return self.s

    def close(self) -> None:
        pass


class Chain:
    """One jumper.barista env, never reset after construction."""

    def __init__(self, bundle: Path, *, scene: str = common.SCENE, seed: int = common.SEED,
                 root_xyzyaw=(0.0, 0.0, 0.11, -math.pi / 2)) -> None:
        common.headless()
        load_fsm_extension()
        import mujoco
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjrl.app_play import AppPlayer, leave_falls_to_the_app, open_app, step_the_world_at
        from mjrl.backend.resolve import resolve
        from mjrl.backend.select import use_backend
        from mjrl.replay import skip_rewards

        import scenes
        import tasks
        from tasks.jumper.common.actuator import CURVE

        self.mujoco, self.torch, self.CURVE = mujoco, torch, CURVE
        use_backend(resolve(backend="native", device="cpu", num_envs=1, training=False))
        self.bundle = Path(bundle)
        self.app = open_app(self.bundle)
        cfg = tasks.load_env_cfg("jumper.barista", play=True)
        cfg.seed = seed
        cfg.sim.mujoco.timestep = 1.0 / common.PHYSICS_HZ
        step_the_world_at(cfg, self.app.control_hz)
        leave_falls_to_the_app(cfg, self.app)
        cfg.episode_length_s = 1.0e6  # one continuous run: no episode end, no reset
        cfg.terminations = {}
        # Keep the seeded startup randomisation M4 also runs under; drop the random spawn,
        # the five-foot carried-arm reset and the claw step events (they would overwrite
        # the app's LF targets every step) and every interval event (pushes).
        drop = {"reset_base", "hold_carried_arm", "gripper_teleop", "claw_hold"}
        cfg.events = {k: v for k, v in cfg.events.items()
                      if k not in drop and getattr(v, "mode", "") != "interval"}
        scenes.apply(cfg, scene)
        cfg.scene.num_envs = 1
        self.cfg = cfg
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        skip_rewards(env)
        self.env = env
        self.wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        env.reset()
        self.model = env.sim.mj_model
        self.robot = env.scene["robot"]
        x, y, z, yaw = root_xyzyaw
        pose = torch.tensor([[x, y, z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]], dtype=torch.float)
        self.robot.write_root_link_pose_to_sim(pose)
        # Boot from the locomotion export's own standing pose (its default_joint_pos).
        contract = json.loads((self.bundle / "locomotion.json").read_text())
        q = torch.tensor([[float(contract["default_joint_pos"][n]) for n in self.robot.joint_names]])
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        env.sim.forward()
        env.scene.update(dt=0.0)
        self.pad = ScriptedPad()
        fsm_mod = sys.modules["mjrl_fsm"]
        real_fsm = fsm_mod.Fsm

        def no_operators(*a, **k):
            k["operators"] = False  # commands in physical units via set_command_axes
            return real_fsm(*a, **k)

        fsm_mod.Fsm = no_operators
        try:
            self.player = AppPlayer(self.app, self.wrapped, pad=self.pad)
        finally:
            fsm_mod.Fsm = real_fsm
        self.fsm = self.player.fsm
        self.cmd: dict[str, float] = {}
        self.servo = self.robot.actuators[0]
        self.dt = float(self.model.opt.timestep)
        self.decimation = int(cfg.decimation)
        self.step_dt = self.dt * self.decimation
        self.joint_names = list(self.servo.target_names)
        self.data = env.sim._datas[0]
        self.beat = "boot"
        self.substep = 0
        self.ctrl_step = 0
        self.t = 0.0
        self._direct = False
        self.external_audit = False
        self.phase_beat: dict[str, str] = {}
        self._install_audit()
        self._index_bodies()
        self.log: list[dict] = []
        self.render_qpos: list[np.ndarray] = []
        self.render_time: list[float] = []
        self.render_stream: list[np.ndarray | None] = []
        self.torque_applied: list[np.ndarray] = []
        self.continuity: list[dict] = []
        self._prev_foot = None
        self._prev_con = None

    # ------------------------------------------------------------------ audit
    def _install_audit(self) -> None:
        servo = self.servo
        servo.enable_audit("a4-chain")
        orig_compute = servo.compute
        orig_drain = servo.drain_audit
        chain = self

        def compute(cmd):
            if not chain.external_audit:
                servo.set_audit_context(phase=chain.beat, substep=chain.substep,
                                        control_step=chain.ctrl_step, time_s=chain.substep * chain.dt)
            return orig_compute(cmd)

        def drain():
            batches = orig_drain()
            chain._ingest(batches)
            return batches

        servo.compute = compute
        servo.drain_audit = drain
        n = len(self.joint_names)
        self.tq: dict[str, dict] = {}
        self._above_el = np.zeros(n)

    def _ingest(self, batches) -> None:
        curve = self.CURVE
        for b in batches:
            app = b.tau_applied_nm.detach().cpu().numpy()[0].astype(float)
            raw = b.tau_raw_nm.detach().cpu().numpy()[0].astype(float)
            qd = b.qd_pre_rad_s.detach().cpu().numpy()[0].astype(float)
            cl = b.curve_limit_nm.detach().cpu().numpy()[0].astype(float)
            el = b.effective_limit_nm.detach().cpu().numpy()[0].astype(float)
            thp = b.thermal_theta_pre.detach().cpu().numpy()[0].astype(float)
            tho = b.thermal_theta_post.detach().cpu().numpy()[0].astype(float)
            self.torque_applied.append(app.astype(np.float32))
            mag = np.abs(app)
            above = mag > curve.continuous_torque
            self._above_el = np.where(above, self._above_el + b.dt_s, 0.0)
            rec = thp + ((app / curve.continuous_torque) ** 2 - thp) * (b.dt_s / curve.thermal_time_constant)
            t_now = self.t_substep()
            beat = self.phase_beat.get(b.phase, self.beat) if self.external_audit else self.beat
            for key in ("whole_run", beat):
                s = self.tq.get(key)
                if s is None:
                    s = self.tq[key] = {
                        "n": 0, "peak_speed": np.zeros(len(app)), "peak_raw": np.zeros(len(app)),
                        "peak_applied": np.zeros(len(app)), "sumsq": np.zeros(len(app)),
                        "min_curve_margin": np.full(len(app), np.inf),
                        "min_eff_margin": np.full(len(app), np.inf), "max_theta": np.zeros(len(app)),
                        "above_count": np.zeros(len(app)), "max_contig": np.zeros(len(app)),
                        "derated": np.zeros(len(app)), "violations": 0, "first_violation_t": None,
                        "max_recompute_err": 0.0, "dt": b.dt_s, "max_contig_at": None,
                        "max_theta_at": None,
                    }
                s["n"] += 1
                s["peak_speed"] = np.maximum(s["peak_speed"], np.abs(qd))
                s["peak_raw"] = np.maximum(s["peak_raw"], np.abs(raw))
                s["peak_applied"] = np.maximum(s["peak_applied"], mag)
                s["sumsq"] += app ** 2
                s["min_curve_margin"] = np.minimum(s["min_curve_margin"], cl - mag)
                s["min_eff_margin"] = np.minimum(s["min_eff_margin"], el - mag)
                theta = np.maximum(thp, tho)
                if theta.max() > s["max_theta"].max():
                    s["max_theta_at"] = [t_now, self.joint_names[int(np.argmax(theta))]]
                s["max_theta"] = np.maximum(s["max_theta"], theta)
                s["above_count"] += above
                if self._above_el.max() > s["max_contig"].max():
                    s["max_contig_at"] = [t_now, self.joint_names[int(np.argmax(self._above_el))]]
                s["max_contig"] = np.maximum(s["max_contig"], self._above_el)
                s["derated"] += thp >= 1.0
                viol = ((cl - mag) < -1e-4) | ((el - mag) < -1e-4)
                if viol.any():
                    s["violations"] += int(viol.sum())
                    if s["first_violation_t"] is None:
                        s["first_violation_t"] = t_now
                s["max_recompute_err"] = max(s["max_recompute_err"], float(np.max(np.abs(rec - tho))))

    def torque_summary(self) -> dict:
        out = {}
        for key, s in self.tq.items():
            worst = int(np.argmax(s["peak_applied"]))
            hot = int(np.argmax(s["max_theta"]))
            out[key] = {
                "substeps": s["n"],
                "peak_applied_nm": float(s["peak_applied"].max()),
                "peak_joint": self.joint_names[worst],
                "peak_raw_nm": float(s["peak_raw"].max()),
                "peak_speed_rad_s": float(s["peak_speed"].max()),
                "min_curve_margin_nm": float(s["min_curve_margin"].min()),
                "min_effective_margin_nm": float(s["min_eff_margin"].min()),
                "max_thermal_theta": float(s["max_theta"].max()),
                "max_thermal_at": s["max_theta_at"],
                "hottest_joint": self.joint_names[hot],
                "max_contiguous_above_1p2_s": float(s["max_contig"].max()),
                "max_contiguous_at": s["max_contig_at"],
                "above_1p2_total_s": float(s["above_count"].max() * s["dt"]),
                "derated_substeps": int(s["derated"].sum()),
                "curve_violations": s["violations"],
                "first_violation_t": s["first_violation_t"],
                "max_thermal_recompute_err": s["max_recompute_err"],
                "rms_applied_max_nm": float(np.sqrt(s["sumsq"] / s["n"]).max()),
                "joints_covered": len(s["peak_applied"]),
                "pass": bool(s["violations"] == 0 and s["max_contig"].max() <= self.CURVE.peak_duration + 1e-12
                             and s["max_theta"].max() < 1.0 and s["max_recompute_err"] <= 1e-5),
            }
        return out

    def thermal(self) -> np.ndarray:
        return self.servo._thermal.detach().cpu().numpy()[0].astype(float).copy()

    # ------------------------------------------------------------------ indexing
    def _index_bodies(self) -> None:
        m, mujoco = self.model, self.mujoco

        def name(kind, i):
            return mujoco.mj_id2name(m, kind, i) or ""

        self.base_body = next(i for i in range(m.nbody)
                              if name(mujoco.mjtObj.mjOBJ_BODY, i).endswith("base_link"))
        self.foot_bodies, self.foot_geoms = {}, {}
        for f in common.FEET:
            bodies = [i for i in range(m.nbody)
                      if re.match(rf"robot/{f}_(foot|palm)", name(mujoco.mjtObj.mjOBJ_BODY, i))]
            tip = [i for i in bodies
                   if name(mujoco.mjtObj.mjOBJ_BODY, i).endswith(("foot_tip_link", "palm_link"))]
            self.foot_bodies[f] = tip[0]
            self.foot_geoms[f] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies
                                  and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self._foot_lookup = {g: f for f, gs in self.foot_geoms.items() for g in gs}
        self.robot_geoms = {g for g in range(m.ngeom)
                            if name(mujoco.mjtObj.mjOBJ_GEOM, g).startswith("robot/")}
        torso = ("base_link", "upper_shell_link", "display_module_link", "tof_sensor_link",
                 "camera_link")
        self.torso_geoms = {g for g in self.robot_geoms
                            if name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])).endswith(torso)}
        self.foot_geom_set = set(self._foot_lookup)
        gname = lambda g: name(mujoco.mjtObj.mjOBJ_GEOM, g)
        self.bar_geoms = {g for g in range(m.ngeom)
                          if gname(g) == "terrain" or "bar_top_ext_" in gname(g)}
        pitcher_body = next((b for b in range(m.nbody)
                             if name(mujoco.mjtObj.mjOBJ_BODY, b).endswith("_pitcher")), None)
        self.pitcher_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pitcher_body}
        self.brewer_geoms = {g for g in range(m.ngeom)
                             if any(t in gname(g) for t in ("server_", "v60_"))}

    # ------------------------------------------------------------------ state
    def t_substep(self) -> float:
        return self.substep * self.dt

    def base(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.base_body].copy(), self.data.xquat[self.base_body].copy()

    def pose2d(self) -> tuple[float, float, float]:
        pos, quat = self.base()
        return float(pos[0]), float(pos[1]), common.yaw_of(quat)

    def joint_q(self) -> dict[str, float]:
        q = self.robot.data.joint_pos[0].detach().cpu().numpy()
        return {n: float(v) for n, v in zip(self.robot.joint_names, q, strict=True)}

    def state_dict(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
            "joint_names": np.asarray(list(self.robot.joint_names)),
            "joint_pos": self.robot.data.joint_pos[0].detach().cpu().numpy().astype(float),
            "joint_vel": self.robot.data.joint_vel[0].detach().cpu().numpy().astype(float),
            "thermal": self.thermal(), "thermal_names": np.asarray(self.joint_names),
            "t": np.float64(self.t_substep()),
        }

    def contacts(self) -> tuple[dict, bool]:
        """Foot contacts, and whether a non-foot robot geom touches the bar (``body_contact``).

        ``torso_on_bar`` (base or shell on the bar) is the fall criterion.
        """
        d = self.data
        inc = {f: False for f in common.FEET}
        body_contact = False
        self._scene_flags = {"pitcher": False, "brewer": False, "off_bar": False, "torso": False}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for g, other in ((g1, g2), (g2, g1)):
                if g in self.robot_geoms and other not in self.robot_geoms:
                    if other in self.pitcher_geoms:
                        self._scene_flags["pitcher"] = True
                    elif other in self.brewer_geoms:
                        self._scene_flags["brewer"] = True
                    elif other not in self.bar_geoms:
                        self._scene_flags["off_bar"] = True
                if g in self._foot_lookup:
                    inc[self._foot_lookup[g]] = True
                elif g in self.robot_geoms and other in self.bar_geoms:
                    body_contact = True
                    if g in self.torso_geoms:
                        self._scene_flags["torso"] = True
        return inc, body_contact

    # ------------------------------------------------------------------ stepping
    def observe(self) -> None:
        """One log row (called at every 5 ms control tick, also during the pour)."""
        pos, quat = self.base()
        feet = {f: self.data.xpos[b].copy() for f, b in self.foot_bodies.items()}
        con, body_contact = self.contacts()
        slip = {}
        if self._prev_foot is not None:
            for f in common.FEET:
                if con[f] and self._prev_con[f]:
                    slip[f] = float(np.linalg.norm(feet[f][:2] - self._prev_foot[f][:2]))
        self._prev_foot, self._prev_con = feet, con
        roll, pitch = common.roll_pitch_of(quat)
        mode = "direct" if self._direct else self.fsm.mode
        self.log.append({
            "t": round(self.t_substep(), 6), "beat": self.beat, "mode": mode,
            "running": bool(self.fsm.running_policy) and not self._direct,
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "yaw": common.yaw_of(quat), "roll": roll, "pitch": pitch, "tilt": common.tilt_of(quat),
            "contacts": sum(con.values()), "body_contact": body_contact,
            "pitcher_contact": self._scene_flags["pitcher"],
            "brewer_contact": self._scene_flags["brewer"],
            "off_bar_contact": self._scene_flags["off_bar"],
            "torso_on_bar": self._scene_flags["torso"],
            "slip_step_m": max(slip.values()) if slip else 0.0,
            "slip_sum_m": sum(slip.values()) if slip else 0.0,
        })

    def after_substep(self) -> None:
        """Bookkeeping after every physics substep that the chain itself drives."""
        self.substep += 1
        if self.substep % RENDER_STRIDE == 0:
            self.render_qpos.append(self.data.qpos.copy())
            self.render_time.append(self.t_substep())
            self.render_stream.append(None)

    def _step_core(self) -> None:
        torch = self.torch
        now = self.player._tick * self.player._step_us
        names = list(self.cmd)
        self.fsm.set_command_axes(names, [float(self.cmd[k]) for k in names], now)
        before = self.substep
        sim = self.env.sim
        real_step = sim.step
        chain = self

        def counted(*a, **k):
            result = real_step(*a, **k)
            chain.after_substep()
            return result

        sim.step = counted
        try:
            with torch.inference_mode():
                if self._direct:
                    self.wrapped.step(self.player._zero)
                else:
                    self.wrapped.step(self.player(None))
        finally:
            del sim.step
        if self.substep - before != self.decimation:
            raise RuntimeError(f"expected {self.decimation} substeps, got {self.substep - before}")
        self.ctrl_step += 1
        self.servo.drain_audit()
        self.observe()

    def set_gains(self, kp: float, kd: float) -> None:
        torch = self.torch
        for act, _ in self.player._gains:
            n = len(act.target_names)
            act.set_gains(slice(None), kp=torch.full((1, n), kp), kd=torch.full((1, n), kd))

    def step_direct(self, target_by_name: dict, kp: float = 10.0, kd: float = 0.5) -> None:
        """Leave the app: joint targets and gains straight to the servos, as run_m4 does."""
        torch = self.torch
        names = list(self.robot.joint_names)
        self.player._targets[0, :] = torch.tensor([float(target_by_name[n]) for n in names])
        self.set_gains(kp, kd)
        self._direct = True
        self._step_core()

    def step(self) -> None:
        self._direct = False
        self._step_core()

    def run(self, seconds: float, beat: str | None = None, each=None) -> None:
        if beat is not None:
            self.beat = beat
        for _ in range(round(seconds / self.step_dt)):
            if each is not None:
                each(self)
            self.step()

    def press(self, buttons=(), hat=(0, 0), hold_s: float = 0.06, beat: str | None = None) -> None:
        self.pad.s.buttons = set(buttons)
        self.pad.s.hat_x, self.pad.s.hat_y = hat
        self.run(hold_s, beat=beat)

    def release(self, s: float = 0.06, beat: str | None = None) -> None:
        self.pad.s.buttons = set()
        self.pad.s.hat_x = self.pad.s.hat_y = 0
        self.run(s, beat=beat)

    def reenter_app(self, gap_substeps: int) -> None:
        """Hand back to the app after a segment it did not tick (MOTION_CHAIN.md Q2)."""
        self.player._tick += gap_substeps // self.decimation
