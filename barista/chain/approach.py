"""pre_align -> pour_start: the scripted, quasi-static six-foot approach (A4).

Simulation only. The supervisor's decision on A4 fixes the walk's end at A1's ``pre_align``
stance (x 0.72 m, 80 mm short of the M4 ``pour_start``, the 6-leg stance >= 20 mm clear of
the pitcher), and the pour to the accepted M4 start. The 80 mm between them cannot be walked
by a policy: the six-leg walking stance's LF palm sits inside the pitcher at the M4 slot
(A1: 29 mm), and the five-foot claw walk overheats RF_J1 or stalls with the arm raised
(MOTION_CHAIN.md Q4). So the approach is a reference trajectory in joint space, computed by
leg IK on a scratch MjData and tracked by the servos at run_m4's own gains (kp 10, kd 0.5):

1. ``blend_home``  - every joint to HOME in place (LF still a foot).
2. ``shift_1``     - the body slides half-way to pour_start over all six planted feet.
3. ``step_<leg>``  - RR, LR, RM, LM, RF each step, one at a time (five feet down), to their
                     HOME footholds *relative to pour_start*.
4. ``shift_2``     - the body slides the rest of the way; five legs are now exactly at HOME.
5. ``lf_lift``     - LF, still planted where it walked to (well clear of the pitcher), lifts
                     straight up, swings to a high claw-open pose above the grip bar, and
                     lowers onto the M4 keyframe-0 pose with the claw open (the M4 start).

Base x/y/yaw errors left by the walk are corrected here (the footholds and the body target
are pour_start's, not the arrival's); what the servos' compliance leaves is absorbed by the
closed-loop pour re-targeting (pour.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

common = sys.modules.get("barista_chain_common")
if common is None:  # loaded by path
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("barista_chain_common", Path(__file__).with_name("common.py"))
    common = _ilu.module_from_spec(_spec)
    sys.modules["barista_chain_common"] = common
    _spec.loader.exec_module(common)

LEG_JOINTS = {
    "LF": ("LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint"),
    "RF": ("RF_J0_joint", "RF_J1_joint", "RF_J2_joint", "RF_J3_joint"),
    "LM": ("LM_J0_joint", "LM_J1_joint", "LM_J2_joint"),
    "RM": ("RM_J0_joint", "RM_J1_joint", "RM_J2_joint"),
    "LR": ("LR_J0_joint", "LR_J1_joint", "LR_J2_joint"),
    "RR": ("RR_J0_joint", "RR_J1_joint", "RR_J2_joint"),
}
STEP_ORDER = ("RR", "LR", "RM", "LM", "RF")
KP, KD = 10.0, 0.5
DURATIONS = {"blend_home": 2.0, "shift": 3.0, "step": 1.2, "step_pause": 0.4,
             "lf_up": 0.8, "lf_raise": 0.8, "lf_high": 1.2, "lf_place": 2.0, "hold": 1.0}
STEP_HEIGHT_M = 0.025
LF_UP_M = 0.045
#: Claw-open LF pose above the grip bar: M4 keyframe 0 with the shoulder pitch raised
#: (kinematic clearance to the pitcher >= 24 mm for the body anywhere in 0.74-0.805 m).
LF_HIGH_J1 = -2.3


class LegIK:
    def __init__(self, model) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        name = lambda kind, i: mujoco.mj_id2name(model, kind, i) or ""
        jid = {name(mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1]: j for j in range(model.njnt)}
        self.jid = jid
        self.site = {leg: next(s for s in range(model.nsite)
                               if name(mujoco.mjtObj.mjOBJ_SITE, s) == f"robot/{leg}")
                     for leg in LEG_JOINTS}
        self.qadr = {n: int(model.jnt_qposadr[j]) for n, j in jid.items()}
        self.vadr = {n: int(model.jnt_dofadr[j]) for n, j in jid.items()}
        self.range = {n: model.jnt_range[j].copy() for n, j in jid.items()}

    def set(self, qpos, base_xyzyaw, joints: dict) -> None:
        d = self.data
        d.qpos[:] = qpos
        x, y, z, yaw = base_xyzyaw
        d.qpos[0:3] = [x, y, z]
        d.qpos[3:7] = common.yaw_quat(yaw)
        for n, v in joints.items():
            d.qpos[self.qadr[n]] = v
        self.mujoco.mj_kinematics(self.model, d)
        self.mujoco.mj_comPos(self.model, d)

    def feet(self) -> dict:
        return {leg: self.data.site_xpos[s].copy() for leg, s in self.site.items()}

    def solve(self, qpos, base_xyzyaw, joints: dict, targets: dict, rest: dict,
              iters: int = 30) -> tuple[dict, float]:
        """Per-leg damped least squares on the foot site, pulled weakly toward ``rest``."""
        q = dict(joints)
        worst = 0.0
        for _ in range(iters):
            self.set(qpos, base_xyzyaw, q)
            worst = 0.0
            for leg, target in targets.items():
                names = LEG_JOINTS[leg]
                site = self.site[leg]
                err = target - self.data.site_xpos[site]
                worst = max(worst, float(np.linalg.norm(err)))
                jacp = np.zeros((3, self.model.nv))
                self.mujoco.mj_jacSite(self.model, self.data, jacp, None, site)
                jac = jacp[:, [self.vadr[n] for n in names]]
                lam = 0.02
                a = np.vstack([jac, lam * np.eye(len(names))])
                b = np.concatenate([err, lam * np.array([rest[n] - q[n] for n in names]) * 0.1])
                step, *_ = np.linalg.lstsq(a, b, rcond=None)
                for n, v in zip(names, step, strict=True):
                    lo, hi = self.range[n]
                    q[n] = float(np.clip(q[n] + v, lo, hi))
            if worst < 1e-5:
                break
        return q, worst


def home_with(home: dict, **overrides) -> dict:
    out = dict(home)
    out.update(overrides)
    return out


def m4_start_pose(home: dict, config: dict, gripper_open: float) -> dict:
    q = dict(home)
    for n, v in zip(config["arm_joint_order"], config["keyframes"][0]["arm_rad"], strict=True):
        q[n] = float(v)
    q["LF_J4_joint"] = gripper_open
    return q


class ApproachPlan:
    """The approach as a function of time, given the measured state at the start of a stage."""

    def __init__(self, model, home: dict, goal_xyzyaw, m4_start: dict) -> None:
        self.ik = LegIK(model)
        self.home = home
        self.goal = np.asarray(goal_xyzyaw, dtype=float)
        self.m4_start = m4_start
        self.lf_high = dict(m4_start)
        self.lf_high["LF_J1_joint"] = LF_HIGH_J1

    def footholds_at(self, qpos, base_xyzyaw) -> dict:
        """HOME footholds for a body at ``base_xyzyaw``."""
        self.ik.set(qpos, base_xyzyaw, self.home)
        return self.ik.feet()

    def stages(self) -> list[tuple[str, float]]:
        out = [("blend_home", DURATIONS["blend_home"]), ("shift_1", DURATIONS["shift"])]
        for leg in STEP_ORDER:
            out.append((f"step_{leg}", DURATIONS["step"] + DURATIONS["step_pause"]))
        out += [("lf_up", DURATIONS["lf_up"]), ("lf_raise", DURATIONS["lf_raise"]),
                ("lf_high", DURATIONS["lf_high"]), ("shift_2", DURATIONS["shift"]),
                ("lf_place", DURATIONS["lf_place"]), ("hold", DURATIONS["hold"])]
        return out


def interp_pose(a, b, s):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = a + (b - a) * s
    out[3] = a[3] + common.wrap(b[3] - a[3]) * s
    return out


def run_approach(chain, model_home: dict, goal_xyzyaw, m4_start: dict, *, record=None,
                 lf_final_fn=None) -> dict:
    """Drive the chain through the approach in direct mode. Returns per-stage metrics."""
    plan = ApproachPlan(chain.model, model_home, goal_xyzyaw, m4_start)
    ik = plan.ik
    dt = chain.step_dt
    report = {"stages": []}

    def ticks(seconds):
        return max(1, round(seconds / dt))

    def measured_base():
        x, y, yaw = chain.pose2d()
        z = float(chain.base()[0][2])
        return np.array([x, y, z, yaw])

    def measured_feet():
        return {leg: chain.data.site_xpos[ik.site[leg]].copy() for leg in LEG_JOINTS}

    def stage_record(name, t0, extra=None):
        x, y, yaw = chain.pose2d()
        rec = {"stage": name, "t0_s": t0, "t1_s": chain.t_substep(), "base_x": x, "base_y": y,
               "base_yaw_deg": math.degrees(yaw)}
        if extra:
            rec.update(extra)
        report["stages"].append(rec)
        if record is not None:
            record(rec)

    # 1. blend every joint to HOME in place
    t0 = chain.t_substep()
    q_start = chain.joint_q()
    n = ticks(DURATIONS["blend_home"])
    for k in range(1, n + 1):
        s = common.smooth(k / n)
        chain.step_direct({j: q_start[j] + (model_home[j] - q_start[j]) * s for j in q_start}, KP, KD)
    stage_record("blend_home", t0)

    qpos0 = chain.data.qpos.copy()
    base_b1 = measured_base()
    feet = measured_feet()
    goal = np.array([plan.goal[0], plan.goal[1], base_b1[2], plan.goal[3]])
    mid = interp_pose(base_b1, goal, 0.5)
    goal_feet = plan.footholds_at(qpos0, goal)
    joints = dict(model_home)

    def track(base_from, base_to, seconds, foot_fn, name, extra_targets=None):
        nonlocal joints
        t_start = chain.t_substep()
        n = ticks(seconds)
        worst = 0.0
        for k in range(1, n + 1):
            s = k / n
            base = interp_pose(base_from, base_to, common.smooth(s))
            targets = foot_fn(s)
            joints, err = ik.solve(qpos0, base, joints, targets, model_home)
            worst = max(worst, err)
            cmd = dict(joints)
            if extra_targets:
                cmd.update(extra_targets)
            chain.step_direct(cmd, KP, KD)
        stage_record(name, t_start, {"ik_worst_residual_mm": 1e3 * worst})

    # 2. body halfway over six planted feet
    track(base_b1, mid, DURATIONS["shift"], lambda s: dict(feet), "shift_1")
    # 3. five single-leg steps to the HOME footholds of pour_start
    for leg in STEP_ORDER:
        start = feet[leg].copy()
        end = goal_feet[leg].copy()

        def swing(s, start=start, end=end, leg=leg):
            out = dict(feet)
            u = min(1.0, s * (DURATIONS["step"] + DURATIONS["step_pause"]) / DURATIONS["step"])
            p = start + (end - start) * common.smooth(u)
            p[2] = start[2] + (end[2] - start[2]) * u + STEP_HEIGHT_M * math.sin(math.pi * u)
            out[leg] = p
            return out

        track(mid, mid, DURATIONS["step"] + DURATIONS["step_pause"], swing, f"step_{leg}")
        feet[leg] = end
    # 4. LF, still planted where it walked to (>= 36 mm from the pitcher), lifts straight up
    #    and swings over to the high claw-open pose while the body is still 40 mm short.
    lf0 = feet["LF"].copy()
    others = {leg: feet[leg] for leg in LEG_JOINTS if leg != "LF"}

    def lift(s):
        out = dict(others)
        p = lf0.copy()
        p[2] += LF_UP_M * common.smooth(s)
        out["LF"] = p
        return out

    track(mid, mid, DURATIONS["lf_up"], lift, "lf_up")
    lf_names = (*LEG_JOINTS["LF"], "LF_J4_joint")
    q_up = {n: joints.get(n, model_home[n]) for n in lf_names}
    q_up["LF_J4_joint"] = model_home["LF_J4_joint"]
    # raise the shoulder first, then fold the rest: the straight joint-space line from the
    # lifted foot to the high pose grazes the pitcher body (measured, 0 mm at mid-path).
    q_raised = dict(q_up)
    q_raised["LF_J1_joint"] = plan.lf_high["LF_J1_joint"]
    held = dict(joints)
    for name_, a, b, secs in (("lf_raise", q_up, q_raised, DURATIONS["lf_raise"]),
                              ("lf_high", q_raised, plan.lf_high, DURATIONS["lf_high"])):
        t_start = chain.t_substep()
        n = ticks(secs)
        for k in range(1, n + 1):
            s = common.smooth(k / n)
            cmd = dict(held)
            for nme in lf_names:
                cmd[nme] = a[nme] + (b[nme] - a[nme]) * s
            chain.step_direct(cmd, KP, KD)
        stage_record(name_, t_start)
    # 5. body the rest of the way on five feet; they land exactly at HOME
    lf_hold = {nme: plan.lf_high[nme] for nme in lf_names}
    track(mid, goal, DURATIONS["shift"], lambda s: dict(others), "shift_2", extra_targets=lf_hold)
    # 6. LF down onto the M4 start, claw open: the grip bar enters the jaws from above. With
    #    ``lf_final_fn`` the arm goes to the pour's own first palm target (closed-loop IK on the
    #    measured base), so the hand-off to pour.py has no step in the LF targets.
    place = dict(m4_start)
    if lf_final_fn is not None:
        place.update(lf_final_fn(chain.data.qpos.copy()))
    report["lf_place_target"] = {nme: float(place[nme]) for nme in lf_names}
    t_start = chain.t_substep()
    n = ticks(DURATIONS["lf_place"])
    for k in range(1, n + 1):
        s = common.smooth(k / n)
        cmd = dict(joints)
        for nme in lf_names:
            cmd[nme] = plan.lf_high[nme] + (place[nme] - plan.lf_high[nme]) * s
        chain.step_direct(cmd, KP, KD)
    stage_record("lf_place", t_start)
    final = dict(joints)
    final.update({nme: place[nme] for nme in lf_names})
    # Every non-LF joint goes exactly to the M4 start (HOME) for the hand-off.
    for nme in final:
        if nme not in lf_names:
            final[nme] = m4_start[nme]
    t_start = chain.t_substep()
    q_now = {nme: final[nme] for nme in final}
    n = ticks(DURATIONS["hold"])
    for _ in range(n):
        chain.step_direct(q_now, KP, KD)
    stage_record("hold", t_start)
    x, y, yaw = chain.pose2d()
    report["final"] = {"base_x": x, "base_y": y, "base_yaw_deg": math.degrees(yaw),
                       "err_x_mm": 1e3 * (x - plan.goal[0]), "err_y_mm": 1e3 * (y - plan.goal[1]),
                       "err_yaw_deg": math.degrees(common.wrap(yaw - plan.goal[3])),
                       "base_z": float(chain.base()[0][2])}
    return report
