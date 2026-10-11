"""Rigid-stance re-targeting of the M4 keyframes (simulation only).

The accepted M4 pour is a list of joint-space keyframes for the LF arm, tuned for a robot
standing exactly at the trajectory's ``base_start``. A robot that arrives a few millimetres or
degrees away from that pose (a walked stance, MOTION_CHAIN.md R1/R2) would replay the same
joint motion shifted with its body, so the spout's spiral would no longer be centred on the
V60 and the pitcher would not land back on its pickup spot.

:class:`StanceRetarget` removes that rigid offset. Once, at the first control tick, it
measures the planar pose of ``base_link`` (x, y, yaw) against ``base_start`` and keeps the
rigid transform ``D`` between them. Every control tick it then asks a small damped
least-squares IK (four LF arm joints, on a scratch ``MjData``) for the arm angles that put
the claw where the keyframe would have put it had the robot stood at ``base_start``:

* while the pitcher is in the claw, the *measured* spout outlet (its offset in the measured
  palm frame) is placed at ``D^-1`` of where the keyframe puts it, with a weak palm-attitude
  term, so the spiral stays centred over the bed in the brew station's frame;
* otherwise (reaching for the grip, setting down, letting go, retreating) the palm frame is
  placed at ``D^-1`` of the keyframe's palm pose.

Set-down servo (optional, ``set_down_servo`` in the trajectory). The pitcher creeps a few
millimetres in the claw while it is carried, so lowering the *claw* onto the pickup spot puts
the *pitcher* down that far off it. During ``lower`` the measured pitcher origin (its current
offset in the measured palm frame) is servoed in x/y onto the pickup spot ``target_m`` (the
keyframe keeps the height), with a small clamped integral on the measured error, ramped in
over ``ramp_s`` so the claw never steps. This servo runs in the nominal stance as well.

With the robot exactly at ``base_start`` (``D`` within :data:`IDENTITY_TOLERANCE`) the
keyframes are returned unchanged, bit for bit, so the nominal replay is the accepted one.
No residual RL, no learned component: the reference keyframes plus inverse kinematics only.
"""

from __future__ import annotations

import math

import numpy as np

ARM_JOINTS = ("LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint")
#: Phases in which the pitcher is held: the spout outlet is re-targeted.
SPOUT_PHASES = frozenset({
    "lift", "lift_settle", "lift_hold", "transit", "pour_in", "spiral", "pour_out", "return",
})
#: After the set-down servo has moved the claw, these palm phases keep that x/y shift so the
#: open claw leaves the pitcher along the accepted path, relative to where it really is.
POST_SET_DOWN_PHASES = frozenset({
    "set_down", "release", "clear_retreat", "retreat", "final_settle",
})
#: Below these the measured stance *is* base_start and the keyframes are not touched.
IDENTITY_TOLERANCE_M = 1e-6
IDENTITY_TOLERANCE_RAD = 1e-6
POSITION_WEIGHT = 1.0
#: Metres of position error one radian of palm-attitude error is worth.
ORIENTATION_WEIGHT_M_PER_RAD = 0.05
#: Pull toward the keyframe angles (metres per radian); keeps the 4-DoF solve well posed.
REGULARISATION_M_PER_RAD = 0.002
IK_ITERATIONS = 8


def planar_pose(position, rotation) -> tuple[float, float, float]:
    """(x, y, yaw) of a body from its world position and 3x3 rotation."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return (
        float(position[0]), float(position[1]),
        math.atan2(float(rotation[1, 0]), float(rotation[0, 0])),
    )


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def rot_log(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if angle < 1e-9:
        return 0.5 * vee
    return vee * (angle / (2.0 * math.sin(angle)))


class StanceOffset:
    """The rigid planar transform from the nominal stance to the measured one."""

    def __init__(self, nominal: tuple[float, float, float],
                 measured: tuple[float, float, float]) -> None:
        self.nominal = tuple(float(v) for v in nominal)
        self.measured = tuple(float(v) for v in measured)
        self.dx = self.measured[0] - self.nominal[0]
        self.dy = self.measured[1] - self.nominal[1]
        self.dyaw = wrap(self.measured[2] - self.nominal[2])
        self._back = rot_z(-self.dyaw)

    @property
    def is_identity(self) -> bool:
        return (
            math.hypot(self.dx, self.dy) <= IDENTITY_TOLERANCE_M
            and abs(self.dyaw) <= IDENTITY_TOLERANCE_RAD
        )

    def to_nominal_point(self, point) -> np.ndarray:
        """``D^-1``: a world point carried by the measured stance back to the nominal one."""
        point = np.asarray(point, dtype=np.float64)
        out = point.copy()
        centre_m = np.array([self.measured[0], self.measured[1]])
        centre_n = np.array([self.nominal[0], self.nominal[1]])
        out[:2] = self._back[:2, :2] @ (point[:2] - centre_m) + centre_n
        return out

    def to_nominal_rotation(self, rotation) -> np.ndarray:
        return self._back @ np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    def as_dict(self) -> dict:
        return {
            "nominal_xy_yaw": list(self.nominal), "measured_xy_yaw": list(self.measured),
            "dx_m": self.dx, "dy_m": self.dy, "dyaw_deg": math.degrees(self.dyaw),
            "identity": self.is_identity,
        }


class StanceRetarget:
    """Damped least-squares IK of the LF arm against the nominal stance (see module doc)."""

    def __init__(self, model, mujoco, base_start: dict,
                 set_down_servo: dict | None = None,
                 spiral_radius_scale: dict | None = None) -> None:
        self.mujoco = mujoco
        self.model = model
        self.scratch = mujoco.MjData(model)

        def one(kind, count: str, suffix: str) -> int:
            hits = [
                index for index in range(getattr(model, count))
                if (mujoco.mj_id2name(model, kind, index) or "").endswith(suffix)
            ]
            if len(hits) != 1:
                raise RuntimeError(f"expected one {suffix!r}, found {hits}")
            return hits[0]

        self.palm = one(mujoco.mjtObj.mjOBJ_BODY, "nbody", "LF_palm_link")
        self.base = one(mujoco.mjtObj.mjOBJ_BODY, "nbody", "base_link")
        self.spout = one(mujoco.mjtObj.mjOBJ_SITE, "nsite", "pitcher_spout_tip")
        self.pitcher = one(mujoco.mjtObj.mjOBJ_BODY, "nbody", "pitcher")
        joints = [one(mujoco.mjtObj.mjOBJ_JOINT, "njnt", f"/{name}") for name in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[joint] for joint in joints])
        self.vadr = np.array([model.jnt_dofadr[joint] for joint in joints])
        self.lower = np.array([model.jnt_range[joint, 0] for joint in joints])
        self.upper = np.array([model.jnt_range[joint, 1] for joint in joints])
        self.nominal = (
            float(base_start["position_m"][0]), float(base_start["position_m"][1]),
            float(base_start["yaw_rad"]),
        )
        self.offset: StanceOffset | None = None
        self.set_down_servo = dict(set_down_servo) if set_down_servo else None
        self.spiral_radius_scale = dict(spiral_radius_scale) if spiral_radius_scale else None
        self.pour_out_start_s: float | None = None
        self.pickup_pitcher_in_palm: np.ndarray | None = None
        self.lower_start_s: float | None = None
        self.max_set_down_shift_m = 0.0
        self.set_down_integral = np.zeros(2)
        self.set_down_shift_xy: np.ndarray | None = None
        self.max_residual_m = 0.0
        self.max_correction_rad = 0.0
        self.solve_count = 0

    # ------------------------------------------------------------------ measurement
    def measure(self, data) -> StanceOffset:
        """Fix ``D`` from the robot's current base pose (call once, at the first tick)."""
        measured = planar_pose(data.xpos[self.base], data.xmat[self.base])
        self.offset = StanceOffset(self.nominal, measured)
        return self.offset

    @property
    def active(self) -> bool:
        return self.offset is not None and not self.offset.is_identity

    # ------------------------------------------------------------------ kinematics
    def _set(self, qpos, arm) -> None:
        self.scratch.qpos[:] = qpos
        self.scratch.qpos[self.qadr] = arm
        self.mujoco.mj_kinematics(self.model, self.scratch)
        self.mujoco.mj_comPos(self.model, self.scratch)

    def _palm(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.scratch.xpos[self.palm].copy(),
            self.scratch.xmat[self.palm].reshape(3, 3).copy(),
        )

    def _jacobian(self, point) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jac(self.model, self.scratch, jacp, jacr, point, self.palm)
        return jacp[:, self.vadr], jacr[:, self.vadr]

    def _solve(self, qpos, arm0, *, point_in_palm, point_target, rot_target,
               orientation_weight: float = ORIENTATION_WEIGHT_M_PER_RAD) -> np.ndarray:
        arm = np.asarray(arm0, dtype=np.float64).copy()
        for _ in range(IK_ITERATIONS):
            self._set(qpos, arm)
            palm_pos, palm_rot = self._palm()
            point = palm_pos + palm_rot @ point_in_palm
            jacp, _ = self._jacobian(point)
            _, jacr = self._jacobian(palm_pos)
            rows = [
                POSITION_WEIGHT * jacp,
                orientation_weight * jacr,
                REGULARISATION_M_PER_RAD * np.eye(len(arm)),
            ]
            errors = [
                POSITION_WEIGHT * (point_target - point),
                orientation_weight * rot_log(rot_target @ palm_rot.T),
                REGULARISATION_M_PER_RAD * (np.asarray(arm0) - arm),
            ]
            step, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(errors), rcond=None)
            arm = np.clip(arm + step, self.lower, self.upper)
        self._set(qpos, arm)
        palm_pos, palm_rot = self._palm()
        residual = float(np.linalg.norm(point_target - (palm_pos + palm_rot @ point_in_palm)))
        self.max_residual_m = max(self.max_residual_m, residual)
        self.max_correction_rad = max(
            self.max_correction_rad, float(np.max(np.abs(arm - np.asarray(arm0))))
        )
        self.solve_count += 1
        return arm

    # ------------------------------------------------------------------ public API
    def correct(self, data, phase: str, arm, time_s: float = 0.0) -> np.ndarray:
        """Arm targets that undo the stance offset for this keyframe ``arm``."""
        arm = np.asarray(arm, dtype=np.float64)
        measured_palm = data.xpos[self.palm].copy()
        measured_rot = data.xmat[self.palm].reshape(3, 3).copy()
        pitcher_in_palm = measured_rot.T @ (data.xpos[self.pitcher] - measured_palm)
        if phase == "lift_hold" and self.pickup_pitcher_in_palm is None:
            self.pickup_pitcher_in_palm = pitcher_in_palm.copy()
        servo = (
            self.set_down_servo is not None and phase == "lower"
            and self.pickup_pitcher_in_palm is not None
        )
        scale = self._radius_scale(phase, time_s)
        post_shift = (
            self.set_down_shift_xy is not None and phase in POST_SET_DOWN_PHASES
        )
        if not self.active and not servo and scale == 1.0 and not post_shift:
            return arm
        assert self.offset is not None
        qpos = data.qpos.copy()
        if servo:
            if self.lower_start_s is None:
                self.lower_start_s = time_s
            ramp = float(self.set_down_servo.get("ramp_s", 1.5))
            fraction = float(np.clip((time_s - self.lower_start_s) / ramp, 0.0, 1.0))
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            self._set(qpos, arm)
            palm_pos, palm_rot = self._palm()
            keyframe_point = self.offset.to_nominal_point(palm_pos + palm_rot @ pitcher_in_palm)
            # The pickup spot is fixed in the world: servo the measured pitcher origin onto
            # it in x/y (the keyframe keeps the height), with a small clamped integral on the
            # measured error so the hanging pitcher lands on it rather than near it.
            spot = np.asarray(self.set_down_servo["target_m"][:2], dtype=np.float64)
            error = spot - np.asarray(data.xpos[self.pitcher][:2], dtype=np.float64)
            dt = float(self.set_down_servo.get("control_dt_s", 0.02))
            clamp = float(self.set_down_servo.get("integral_clamp_m", 0.015))
            self.set_down_integral = np.clip(
                self.set_down_integral + float(self.set_down_servo.get("ki", 2.0)) * dt * error,
                -clamp, clamp,
            )
            shifted = keyframe_point.copy()
            self.set_down_shift_xy = fraction * (spot - keyframe_point[:2] + self.set_down_integral)
            shifted[:2] += self.set_down_shift_xy
            self.max_set_down_shift_m = max(
                self.max_set_down_shift_m, float(np.linalg.norm(spot - keyframe_point[:2]))
            )
            return self._solve(
                qpos, arm, point_in_palm=pitcher_in_palm, point_target=shifted,
                rot_target=self.offset.to_nominal_rotation(palm_rot),
                orientation_weight=float(self.set_down_servo.get("orientation_weight", 0.01)),
            )
        if phase in SPOUT_PHASES:
            point_in_palm = measured_rot.T @ (data.site_xpos[self.spout] - measured_palm)
        else:
            point_in_palm = np.zeros(3)
        # Where the keyframe puts the point and the palm on the measured body, carried back
        # to the nominal stance: the station-frame target the accepted keyframe stands for.
        self._set(qpos, arm)
        palm_pos, palm_rot = self._palm()
        point_target = self.offset.to_nominal_point(palm_pos + palm_rot @ point_in_palm)
        if post_shift:
            point_target[:2] += self.set_down_shift_xy
        if scale != 1.0:
            centre = np.asarray(self.spiral_radius_scale["centre_m"][:2], dtype=np.float64)
            point_target[:2] = centre + scale * (point_target[:2] - centre)
        rot_target = self.offset.to_nominal_rotation(palm_rot)
        return self._solve(
            qpos, arm, point_in_palm=point_in_palm, point_target=point_target,
            rot_target=rot_target,
        )

    def _radius_scale(self, phase: str, time_s: float) -> float:
        """Spiral radius scale about the bed centre: ``scale`` while pouring, eased back to 1
        over the first ``blend_out_s`` of ``pour_out`` so the claw never steps."""
        if self.spiral_radius_scale is None:
            return 1.0
        scale = float(self.spiral_radius_scale["scale"])
        if phase == "spiral":
            return scale
        if phase == "pour_out":
            if self.pour_out_start_s is None:
                self.pour_out_start_s = time_s
            blend = float(self.spiral_radius_scale.get("blend_out_s", 0.5))
            fraction = float(np.clip((time_s - self.pour_out_start_s) / blend, 0.0, 1.0))
            return scale + (1.0 - scale) * fraction * fraction * (3.0 - 2.0 * fraction)
        return 1.0

    def report(self) -> dict:
        return {
            "method": "rigid planar stance offset D measured at the first control tick; "
                      "damped least-squares IK on the LF arm puts the held spout outlet (or "
                      "the palm when nothing is held) at D^-1 of the keyframe's pose",
            "stance_offset": self.offset.as_dict() if self.offset is not None else None,
            "active": self.active,
            "solve_count": self.solve_count,
            "maximum_ik_residual_m": self.max_residual_m,
            "maximum_joint_correction_rad": self.max_correction_rad,
            "set_down_servo": self.set_down_servo,
            "spiral_radius_scale": self.spiral_radius_scale,
            "pickup_pitcher_in_palm_m": (
                None if self.pickup_pitcher_in_palm is None
                else self.pickup_pitcher_in_palm.tolist()
            ),
            "maximum_set_down_creep_compensation_m": self.max_set_down_shift_m,
            "post_set_down_palm_shift_xy_m": (
                None if self.set_down_shift_xy is None else self.set_down_shift_xy.tolist()
            ),
        }
