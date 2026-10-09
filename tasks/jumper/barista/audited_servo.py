"""Plain-MuJoCo servo-curve actuation with immutable substep audit rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tasks.jumper.common.actuator import CURVE
from tasks.jumper.common.constants import DAMPING, EFFORT_LIMIT, STIFFNESS


@dataclass(frozen=True)
class AuditSample:
    """One joint at one physics substep, before integration."""

    q_pre_rad: float
    qd_pre_rad_s: float
    tau_raw_nm: float
    tau_applied_nm: float
    curve_limit_nm: float
    effective_limit_nm: float
    thermal_theta_pre: float
    thermal_theta_post: float


class AuditedServo:
    """The repository servo law in NumPy, including its thermal recurrence."""

    def __init__(self, model, mujoco, joint_names: list[str]) -> None:
        self.names = joint_names
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            raise ValueError("all audited joint names must exist in the compiled model")
        self.qadr = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
        self.dadr = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
        self.dt = float(model.opt.timestep)
        self.thermal = np.zeros(len(joint_names), dtype=np.float64)

    @staticmethod
    def curve_limit(speed: np.ndarray) -> np.ndarray:
        """Measured output-shaft torque-speed limit in N m."""
        magnitude = np.abs(speed)
        limit = CURVE.plateau_torque * np.exp(
            -np.clip(magnitude - CURVE.corner_speed, 0.0, None) / CURVE.decay_speed
        )
        return np.where(magnitude >= CURVE.cutoff_speed, 0.0, limit)

    def apply(self, data, target: np.ndarray) -> list[AuditSample]:
        """Apply one substep of bounded PD torque and return pre-step audit samples."""
        q = data.qpos[self.qadr].copy()
        qd = data.qvel[self.dadr].copy()
        raw = STIFFNESS * (target - q) - DAMPING * qd
        curve = self.curve_limit(qd)
        thermal_pre = self.thermal.copy()
        force_cap = np.where(
            thermal_pre >= 1.0,
            np.minimum(EFFORT_LIMIT, CURVE.continuous_torque),
            EFFORT_LIMIT,
        )
        effective = np.minimum(curve, force_cap)
        applied = np.clip(raw, -effective, effective)
        self.thermal += (
            (applied / CURVE.continuous_torque) ** 2 - self.thermal
        ) * (self.dt / CURVE.thermal_time_constant)
        np.maximum(self.thermal, 0.0, out=self.thermal)
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[self.dadr] = applied
        return [
            AuditSample(*values)
            for values in zip(
                q,
                qd,
                raw,
                applied,
                curve,
                effective,
                thermal_pre,
                self.thermal,
                strict=True,
            )
        ]