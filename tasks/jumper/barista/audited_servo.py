"""Barista-only servo audit at the runtime actuator's pre-integration boundary.

The task swaps this subclass in for the common actuator configuration. The
torque itself still comes from :meth:`ServoCurveActuator.compute`; this class
only snapshots its inputs, limits, output, and thermal state around that call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.actuator.pd_actuator import pd_torque

from tasks.jumper.common.actuator import (
    ServoCurveActuator,
    ServoCurveActuatorCfg,
    servo_torque_limit,
)


@dataclass(frozen=True)
class AuditBatch:
    """One immutable snapshot for every joint in one environment substep."""

    run_id: str
    phase: str
    substep: int
    control_step: int
    time_s: float
    dt_s: float
    q_pre_rad: torch.Tensor
    qd_pre_rad_s: torch.Tensor
    tau_raw_nm: torch.Tensor
    tau_applied_nm: torch.Tensor
    curve_limit_nm: torch.Tensor
    effective_limit_nm: torch.Tensor
    thermal_theta_pre: torch.Tensor
    thermal_theta_post: torch.Tensor


@dataclass(kw_only=True)
class AuditedServoCurveActuatorCfg(ServoCurveActuatorCfg):
    """Build the task-local audited subclass with the common servo parameters."""

    def build(self, entity, target_ids: list[int], target_names: list[str]):
        return AuditedServoCurveActuator(self, entity, target_ids, target_names)


class AuditedServoCurveActuator(ServoCurveActuator):
    """Preserve the common compute path and capture its exact substep boundary."""

    def __init__(self, cfg, entity, target_ids, target_names) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._audit_enabled = False
        self._audit_run_id = ""
        self._audit_context: tuple[str, int, int, float] | None = None
        self._audit_buffer: list[AuditBatch] = []

    def enable_audit(self, run_id: str) -> None:
        """Start a cold, empty audit stream after task construction and reset."""
        self._audit_enabled = True
        self._audit_run_id = run_id
        self._audit_context = None
        self._audit_buffer.clear()

    def set_audit_context(
        self, *, phase: str, substep: int, control_step: int, time_s: float
    ) -> None:
        """Name the next runtime substep before ``scene.write_data_to_sim``."""
        self._audit_context = (phase, substep, control_step, time_s)

    def drain_audit(self) -> tuple[AuditBatch, ...]:
        """Drain snapshots after ``sim.step`` without changing recorded tensors."""
        drained = tuple(self._audit_buffer)
        self._audit_buffer.clear()
        return drained

    def compute(self, cmd) -> torch.Tensor:
        """Call the common implementation once, snapshotting around that call."""
        if not self._audit_enabled:
            return super().compute(cmd)
        if self._audit_context is None:
            raise RuntimeError("audit context must be set before each simulated substep")
        assert self._thermal is not None
        assert self.force_limit is not None
        assert self.continuous_torque is not None
        assert self.plateau_torque is not None
        assert self.corner_speed is not None
        assert self.decay_speed is not None
        assert self.cutoff_speed is not None
        assert self.stiffness is not None
        assert self.damping is not None

        q_pre = cmd.pos.detach().clone()
        qd_pre = cmd.vel.detach().clone()
        raw = pd_torque(self.stiffness, self.damping, cmd).detach().clone()
        thermal_pre = self._thermal.detach().clone()
        curve = servo_torque_limit(
            cmd.vel,
            self.plateau_torque,
            self.corner_speed,
            self.decay_speed,
            self.cutoff_speed,
        ).detach().clone()
        force_cap = torch.where(
            thermal_pre >= 1.0,
            torch.minimum(self.force_limit, self.continuous_torque),
            self.force_limit,
        )
        effective = torch.minimum(curve, force_cap).detach().clone()

        # This is the load-bearing operation: clamp and heat integration remain
        # exactly the common runtime implementation, including one heat update.
        applied = super().compute(cmd)
        phase, substep, control_step, time_s = self._audit_context
        self._audit_buffer.append(
            AuditBatch(
                run_id=self._audit_run_id,
                phase=phase,
                substep=substep,
                control_step=control_step,
                time_s=time_s,
                dt_s=self._substep_dt,
                q_pre_rad=q_pre,
                qd_pre_rad_s=qd_pre,
                tau_raw_nm=raw,
                tau_applied_nm=applied.detach().clone(),
                curve_limit_nm=curve,
                effective_limit_nm=effective,
                thermal_theta_pre=thermal_pre,
                thermal_theta_post=self._thermal.detach().clone(),
            )
        )
        self._audit_context = None
        return applied


__all__ = ["AuditBatch", "AuditedServoCurveActuator", "AuditedServoCurveActuatorCfg"]